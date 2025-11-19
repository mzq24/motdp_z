import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Callable
from collections import defaultdict
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from model.transformer_for_diffusion import TransformerForDiffusion, LowdimMaskGenerator
from model.interfuser_bev_encoder import InterfuserBEVEncoder
from model.interfuser_bev_encoder import load_lidar_submodules
import os
from collections import OrderedDict
from collections import deque
from PIL import Image

import sys
bagel_path = "/root/z_projects/code/Bagel"
if bagel_path not in sys.path:
    sys.path.insert(0, bagel_path)
from modeling.bagel import SiglipVisionConfig, SiglipVisionModel
from data.data_utils import pil_img2rgb, patchify, add_special_tokens
from data.transforms import ImageTransform
from modeling.qwen2 import Qwen2Tokenizer

VLMDriveBackbone = None
VLM_AVAILABLE = False

class PIDController(object):
	def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
		self._K_P = K_P
		self._K_I = K_I
		self._K_D = K_D

		self._window = deque([0 for _ in range(n)], maxlen=n)
		self._max = 0.0
		self._min = 0.0

	def step(self, error):
		self._window.append(error)
		self._max = max(self._max, abs(error))
		self._min = -abs(self._max)

		if len(self._window) >= 2:
			integral = np.mean(self._window)
			derivative = (self._window[-1] - self._window[-2])
		else:
			integral = 0.0
			derivative = 0.0

		return self._K_P * error + self._K_I * integral + self._K_D * derivative

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    # unnormalize from [-1, 1] to [0, 1]
    ndata = (ndata + 1) / 2
    # unnormalize to original range
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

class DiffusionDiTCarlaPolicy(nn.Module):
    def __init__(self, config: Dict, action_stats: Optional[Dict[str, torch.Tensor]] = None, 
                 device: str = 'cuda', use_vlm_features: bool = True):
        super().__init__()
        
        # config
        self.config = config
        self.device = device
        self.use_vlm_features = use_vlm_features
        policy_cfg = config['policy']
        noise_scheduler_cfg = config['noise_scheduler']

        obs_as_global_cond = policy_cfg.get('obs_as_global_cond', True)
        self.obs_as_global_cond = obs_as_global_cond
        shape_meta = config['shape_meta']
        action_shape = shape_meta['action']['shape']
        action_dim = action_shape[0]
        
        # Action normalization settings
        self.enable_action_normalization = config.get('enable_action_normalization', False)
        self.action_stats = action_stats
        if self.enable_action_normalization and self.action_stats is not None:
            print(f"✓ Action normalization enabled with stats:")
            print(f"  Action min: {self.action_stats['min']}")
            print(f"  Action max: {self.action_stats['max']}")
        else:
            print("⚠ Action normalization disabled")
            self.action_stats = None
        
        self.n_obs_steps = policy_cfg.get('n_obs_steps', config.get('obs_horizon', 1))
        
        # load lidar bev encoder
        obs_encoder = self.load_lidar_bev_encoder()
        self.obs_encoder = obs_encoder

        # load image encoder
        self.vit_model, self.vit_transform, self.vit_config = self.load_vit_model()
        self.vit_connector = nn.Linear(self.vit_config.hidden_size, 256)

        # TODO load vlm and vlm encoder model）
        self.vlm_backbone = None
        self.feature_encoder = None
        
        if VLM_AVAILABLE and VLMDriveBackbone is not None:
            try:
                vlm_device = 'cpu'  
                self.vlm_backbone = VLMDriveBackbone(
                    model_type='qwen',
                    checkpoint_path='Qwen/Qwen2.5-VL-3B-Instruct',
                    device=vlm_device
                )
                print("✓ VLM backbone initialized successfully on CPU")
            except Exception as e:
                print(f"⚠ VLM backbone initialization failed: {e}")
                self.vlm_backbone = None
        else:
            print("⚠ VLM backbone not available, using simulated features")
        self.feature_encoder = nn.Linear(2560, 1536)
        if not self.use_vlm_features:
            self.feature_encoder.eval()
            self._init_loaded_vlm_features()


        # create diffusion model
        # TCP模型输出特征维度（j_ctrl）为256，修改相应维度
        obs_feature_dim = 256  
        
        # Optional GroupNorm for j_ctrl features (recommended for training stability)
        # self.use_j_ctrl_norm = policy_cfg.get('use_j_ctrl_norm', False)
        # if self.use_j_ctrl_norm:
        #     self.j_ctrl_norm = nn.GroupNorm(num_groups=8, num_channels=256)
        #     print("✓ GroupNorm enabled for j_ctrl features")  

        self.model = TransformerForDiffusion(
            input_dim=policy_cfg.get('input_dim', 2),
            output_dim=policy_cfg.get('output_dim', 2),
            horizon=policy_cfg.get('horizon', 16),
            n_obs_steps=self.n_obs_steps,  
            cond_dim=256,   
            n_layer=policy_cfg.get('n_layer', 8),
            n_head=policy_cfg.get('n_head', 8),
            n_emb=policy_cfg.get('n_emb', 512),
            p_drop_emb=policy_cfg.get('p_drop_emb', 0.1),
            p_drop_attn=policy_cfg.get('p_drop_attn', 0.1),
            causal_attn=policy_cfg.get('causal_attn', True),
            obs_as_cond=obs_as_global_cond,
            n_cond_layers=policy_cfg.get('n_cond_layers', 4)
        )
        
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_cfg.get('num_diffusion_steps', 100),
            beta_start=noise_scheduler_cfg.get('beta_start', 0.0001),
            beta_end=noise_scheduler_cfg.get('beta_end', 0.02),
            beta_schedule=noise_scheduler_cfg.get('beta_schedule', "squaredcos_cap_v2"),
            clip_sample=noise_scheduler_cfg.get('clip_sample', False),
            prediction_type=noise_scheduler_cfg.get('prediction_type', "epsilon"),
        )

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_global_cond) else obs_feature_dim,
            max_n_obs_steps=self.n_obs_steps,  
            fix_obs_steps=True,
            action_visible=False
        )

        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.horizon = policy_cfg.get('horizon', 16)
        self.n_action_steps = policy_cfg.get('action_horizon', 8)
        self.num_inference_steps = policy_cfg.get('num_inference_steps', 100)

        # Controller - Initialize PID controllers with config parameters
        control_cfg = config.get('controller', {})
        self.turn_controller = PIDController(
            K_P=control_cfg.get('turn_KP', 0.75), 
            K_I=control_cfg.get('turn_KI', 0.75), 
            K_D=control_cfg.get('turn_KD', 0.3), 
            n=control_cfg.get('turn_n', 40)
        )
        self.speed_controller = PIDController(
            K_P=control_cfg.get('speed_KP', 5.0),
            K_I=control_cfg.get('speed_KI', 0.5),
            K_D=control_cfg.get('speed_KD', 1.0),
            n=control_cfg.get('speed_n', 40)
        )
        
        # Store config for later use in control_pid
        self.config = config
        print(f"✓ PID controllers initialized")
        print(f"  - Turn controller: KP={control_cfg.get('turn_KP', 0.75)}, KI={control_cfg.get('turn_KI', 0.75)}, KD={control_cfg.get('turn_KD', 0.3)}")
        print(f"  - Speed controller: KP={control_cfg.get('speed_KP', 5.0)}, KI={control_cfg.get('speed_KI', 0.5)}, KD={control_cfg.get('speed_KD', 1.0)}")

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Normalize action from original range to [-1, 1]
        Args:
            action: tensor of shape (..., action_dim)
        Returns:
            normalized action in range [-1, 1]
        """
        if not self.enable_action_normalization or self.action_stats is None:
            return action
        
        device = action.device
        action_min = self.action_stats['min'].to(device)
        action_max = self.action_stats['max'].to(device)
        
        # Normalize to [0, 1]
        normalized = (action - action_min) / (action_max - action_min + 1e-8)
        # Normalize to [-1, 1]
        normalized = normalized * 2 - 1
        return normalized

    def unnormalize_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize action from [-1, 1] back to original range
        Args:
            normalized_action: tensor of shape (..., action_dim) in range [-1, 1]
        Returns:
            action in original range
        """
        if not self.enable_action_normalization or self.action_stats is None:
            return normalized_action
        
        device = normalized_action.device
        action_min = self.action_stats['min'].to(device)
        action_max = self.action_stats['max'].to(device)
        
        # Unnormalize from [-1, 1] to [0, 1]
        unnormalized = (normalized_action + 1) / 2
        # Unnormalize to original range
        unnormalized = unnormalized * (action_max - action_min) + action_min
        return unnormalized

    def extract_tcp_features(self, obs_dict, return_attention=False):
        """
        使用InterfuserBEVEncoder和VIT提取特征
        支持多种输入模式。
        
        Args:
            obs_dict: 观测字典，应包含：
                - 'lidar_token', 'lidar_token_global': 预处理的BEV特征
                - 'lidar_bev': 原始BEV图像
                - 'image': 原始RGB图像
                - 'speed', 'target_point', 'next_command', 'heading': 状态信息
            return_attention: 是否返回attention map
            
        Returns:
            j_ctrl特征 (B, 256) 或 (j_ctrl特征, attention_map)
        """
        #try:
        # 准备状态信息
        speed = obs_dict['speed'].to(dtype=torch.float32).view(-1,1) / 12.
        target_point = obs_dict['target_point'].to(dtype=torch.float32)
        command = obs_dict['next_command'].to(dtype=torch.float32)
        heading = obs_dict.get('heading', torch.zeros_like(speed)).to(dtype=torch.float32).view(-1, 1) # (B, 2)
        state = torch.cat([speed, target_point, command, heading], 1).to(self.device)
        
        # 提取图像特征
        # img_feature = self.extract_vit_feature(obs_dict['image'])

        use_precomputed_lidar = 'lidar_token' in obs_dict and 'lidar_token_global' in obs_dict
        
        if use_precomputed_lidar:
            # 模式1: 使用预处理好的BEV特征
            lidar_token = obs_dict['lidar_token'].to(device=self.device, dtype=torch.float32)
            lidar_token_global = obs_dict['lidar_token_global'].to(device=self.device, dtype=torch.float32)
            
            j_ctrl, attention_map = self.obs_encoder(
                state=state,
                lidar_token=lidar_token,
                lidar_token_global=lidar_token_global,
                normalize=True,
                return_attention=True
            )
        else:
            # 模式2: 使用原始lidar_bev图像
            if 'lidar_bev' not in obs_dict:
                raise KeyError("Neither pre-computed LiDAR features nor raw BEV images found in obs_dict")
            
            lidar_bev_img = obs_dict['lidar_bev'].to(device=self.device, dtype=torch.float32)
            
            j_ctrl, attention_map = self.obs_encoder(
                image=lidar_bev_img,
                state=state,
                normalize=True,
                return_attention=True
            )
        
        if return_attention:
            return j_ctrl, attention_map
        else:
            return j_ctrl
                
        # except KeyError as e:
        #     raise KeyError(f"Missing required field in obs_dict for feature extraction: {e}")
        # except Exception as e:
        #     import traceback
        #     traceback.print_exc()
        #     raise RuntimeError(f"Error in feature extraction: {e}")

    def extract_vit_feature(self, image_tensor):
    
        if image_tensor.ndim != 4:
            raise ValueError(f"Expected a 4D tensor (B, C, H, W), but got shape {image_tensor.shape}")

        device = self.device
        image_tensor = image_tensor.to(device)
        B, C, H, W = image_tensor.shape

        vit_patch_size = 14
        vit_max_num_patch_per_side = 70
        
        # Get position IDs
        vit_position_ids = self.get_flattened_position_ids_extrapolate(H, W, vit_patch_size, 
            max_num_patches_per_side=vit_max_num_patch_per_side).to(device)
        
        # Patchify image
        vit_tokens = self.patchify(image_tensor, vit_patch_size)    # (B, L, patch_dim)
        
        # Prepare inputs for the model
        # The model expects packed sequences, but here we have only one sequence.
        num_img_tokens = vit_tokens.shape[1] # L from (N, L, patch_dim)
        packed_vit_tokens = vit_tokens.reshape(B * num_img_tokens, -1)
        
        vit_token_seqlens = torch.tensor([num_img_tokens] * B, dtype=torch.int, device=device)
        packed_vit_position_ids = vit_position_ids.unsqueeze(0).repeat(B, 1).reshape(B * num_img_tokens)

        # Compute cumulative sequence lengths
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0)).to(torch.int32)
        max_seqlen = num_img_tokens
        
        # Extract features
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                packed_vit_token_embed = self.vit_model(
                    packed_pixel_values=packed_vit_tokens, 
                    packed_flattened_position_ids=packed_vit_position_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
        packed_vit_token_embed = self.vit_connector(packed_vit_token_embed)
        feature_dim = packed_vit_token_embed.shape[-1]
        batched_embed = packed_vit_token_embed.reshape(B, num_img_tokens, feature_dim)

        return batched_embed

    @staticmethod
    def get_flattened_position_ids_extrapolate(img_h, img_w, patch_size, max_num_patches_per_side):
        """Generate position IDs using extrapolation method"""
        num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
        coords_h = torch.arange(0, num_patches_h)
        coords_w = torch.arange(0, num_patches_w)
        pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
        return pos_ids

    @staticmethod
    def patchify(imgs, patch_size):
        """
        Convert a batch of images into patches.
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 * 3)
        """
        p = patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p * p * 3))
        return x

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        batch: {
            # 选项1: 预处理好的特征（推荐）
            'lidar_token': (B, obs_horizon, seq_len, 512) - 预处理的空间特征
            'lidar_token_global': (B, obs_horizon, 1, 512) - 预处理的全局特征
            
            # 选项2: 原始图像（兼容模式）
            'lidar_bev': (B, obs_horizon, 3, 448, 448) - LiDAR BEV图像
            
            # 其他必需字段
            'agent_pos': (B, obs_horizon, 2)
            'next_command': (B, obs_horizon, 6)
            'speed': (B, obs_horizon)
            'target_point': (B, obs_horizon, 2)
        }
        """
        device = next(self.parameters()).device
        nobs = {}
        
        # 支持预处理特征和原始图像两种模式
        carla_fields = ['lidar_token', 'lidar_token_global', 'lidar_bev', 'next_command', 'speed', 'target_point', 'agent_pos', 'heading']
        for field in carla_fields:
            if field in batch:
                if field in ['lidar_bev', 'lidar_token', 'lidar_token_global']:
                    nobs[field] = batch[field].to(device=device, dtype=torch.float32)
                else:
                    nobs[field] = batch[field].to(device)

        raw_agent_pos = batch['agent_pos'].to(device)
        if 'vqa' not in batch:
            assert self.use_vlm_features == False, "VLM features expected but 'vqa' not found in batch"
        if not self.use_vlm_features:
            vl_features, vl_mask = self.generate_simulated_vlm_outputs(raw_agent_pos.shape[0], device)
            batch['vqa'] = vl_features
        # (B, horizon, 2)
        To = self.n_obs_steps
        nactions = raw_agent_pos
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        cond = None
        
        # Normalize trajectory for training
        trajectory = nactions.float()
        if self.enable_action_normalization and self.action_stats is not None:
            trajectory = self.normalize_action(trajectory)
            if torch.isnan(trajectory).any() or torch.isinf(trajectory).any():
                print("Warning: NaN or Inf detected in normalized trajectory")
                trajectory = torch.nan_to_num(trajectory, nan=0.0, posinf=1.0, neginf=-1.0)

        if self.obs_as_global_cond:
            obs_features_list = []
            for t in range(To):
                this_step_nobs = dict_apply(nobs, lambda x: x[:, t, ...])
                step_features = self.extract_tcp_features(this_step_nobs)  # (B, feature_dim)
                obs_features_list.append(step_features)
            
            # 堆叠所有时间步的特征: (B, To, feature_dim)
            cond = torch.stack(obs_features_list, dim=1).float()  
            # add vit features
            if 'image' in batch:
                batch_image = batch['image'][:,-1,:,:,:]                # (B, num_img, C, H, W)
                vit_features = self.extract_vit_feature(batch_image)    # (B, num_token, feat_dim)
                cond = torch.concatenate([cond, vit_features], dim=1)       # (B, To + num_token, feature_dim)
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.extract_tcp_features(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)

        # generate impainting mask
        # condition_mask = self.mask_generator(trajectory.shape)
        condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.get('num_train_timesteps', 100), 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        vqa = batch.get('vqa', None)
        if vqa is None:
            # 如果vqa为None或不存在，使用初始化中生成的模板
            vl_features, vl_mask = self.generate_simulated_vlm_outputs(
                batch_size=noisy_trajectory.shape[0], 
                device=device,
                max_seq_len=None
            )
        else:
            vl_features = vqa.to(device=device, dtype=torch.float32)  # (B, seq_len, feat_dim)
            vl_mask = torch.ones(vl_features.shape[:2], dtype=torch.bool, device=device)
        vl_embeds = self.feature_encoder(vl_features)
        pred = self.model(noisy_trajectory, timesteps, vl_embeds, cond, vl_mask=vl_mask)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        
        if loss.shape[-1] > 2:
            loss = loss[..., :2]  
            
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
    
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None, vl_features=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)
        
        # Use provided vl_features or generate simulated ones
        if vl_features is None:
            vl_features, vl_mask = self.generate_simulated_vlm_outputs(trajectory.shape[0], trajectory.device)
        else:
            # vl_features provided, create mask for valid positions
            vl_mask = torch.ones(vl_features.shape[:2], dtype=torch.bool, device=vl_features.device)
        
        vl_embeds = self.feature_encoder(vl_features)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, vl_embeds, cond, vl_mask=vl_mask)


            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        nobs = dict_apply(obs_dict, lambda x: x.to(device))

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_global_cond:
            obs_features_list = []
            for t in range(To):
                this_step_nobs = dict_apply(nobs, lambda x: x[:, t, ...])
                step_features = self.extract_tcp_features(this_step_nobs)  # (B, feature_dim)
                obs_features_list.append(step_features)
            cond = torch.stack(obs_features_list, dim=1)
            shape = (B, T, Da)
           
            cond_data = torch.zeros(size=shape, device=device, dtype=torch.float32)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            obs_features_list = []
            for t in range(To):
                this_step_nobs = dict_apply(nobs, lambda x: x[:, t, ...])
                step_features = self.extract_tcp_features(this_step_nobs)  # (B, feature_dim)
                obs_features_list.append(step_features)
            
            # 堆叠所有时间步的特征: (B, To, feature_dim)
            nobs_features = torch.stack(obs_features_list, dim=1)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=torch.float32)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        # Extract VQA features from obs_dict if available
        if 'vqa' not in nobs or nobs['vqa'] is None:
            # 如果vqa为None或不存在，使用初始化中生成的模板
            vl_feat, _ = self.generate_simulated_vlm_outputs(
                batch_size=B, 
                device=device,
                max_seq_len=None
            )
        else:
            vl_feat = nobs['vqa'].to(dtype=torch.float32)
        
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            vl_features=vl_feat
            )
        
        naction_pred = nsample[...,:Da]
        
        # Clamp normalized predictions to [-1, 1] to prevent extreme values
        naction_pred = torch.clamp(naction_pred, -1.0, 1.0)
        
        # Unnormalize action predictions back to original range
        if self.enable_action_normalization and self.action_stats is not None:
            naction_pred = self.unnormalize_action(naction_pred)
            
            # Additional safety clamp after unnormalization to prevent extreme outliers
            device = naction_pred.device
            action_min = self.action_stats['min'].to(device)
            action_max = self.action_stats['max'].to(device)
            
            # Expand to reasonable bounds (20% beyond training range)
            safety_margin = 0.2
            range_size = action_max - action_min
            expanded_min = action_min - safety_margin * range_size
            expanded_max = action_max + safety_margin * range_size
            
            naction_pred = torch.clamp(naction_pred, expanded_min, expanded_max)
        
        action_pred = naction_pred.detach().cpu().numpy()
        # 直接返回整个预测序列，因为horizon=action_horizon
        action = action_pred
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    def extract_attention_map(self, obs_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        提取最后一帧观测的attention map
        
        Args:
            obs_dict: 观测字典
            
        Returns:
            attention_map: (H, W) numpy数组
        """
        device = next(self.parameters()).device
        nobs = dict_apply(obs_dict, lambda x: x.to(device))
        
        # 获取最后一帧观测
        last_obs = dict_apply(nobs, lambda x: x[:, -1, ...])
        
        # 提取特征和attention map
        with torch.no_grad():
            _, attention_map = self.extract_tcp_features(last_obs, return_attention=True)
        
        # 转换为numpy
        attention_map_np = attention_map[0].cpu().numpy()  # (H, W)
        
        return attention_map_np

    def predict_action_with_steps(self, obs_dict: Dict[str, torch.Tensor]):
        """
        预测动作并返回去噪过程的中间步骤
        
        Returns:
            result: 预测结果字典
            denoising_steps: 去噪过程中的轨迹列表
        """
        device = next(self.parameters()).device
        nobs = dict_apply(obs_dict, lambda x: x.to(device))

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # 准备条件
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_global_cond:
            obs_features_list = []
            for t in range(To):
                this_step_nobs = dict_apply(nobs, lambda x: x[:, t, ...])
                step_features = self.extract_tcp_features(this_step_nobs)
                obs_features_list.append(step_features)
            cond = torch.stack(obs_features_list, dim=1)
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=torch.float32)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            obs_features_list = []
            for t in range(To):
                this_step_nobs = dict_apply(nobs, lambda x: x[:, t, ...])
                step_features = self.extract_tcp_features(this_step_nobs)
                obs_features_list.append(step_features)
            nobs_features = torch.stack(obs_features_list, dim=1)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=torch.float32)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # 运行采样并捕获中间步骤
        # Extract VQA features from obs_dict if available
        if 'vqa' not in nobs or nobs['vqa'] is None:
            # 如果vqa为None或不存在，使用初始化中生成的模板
            vl_feat, _ = self.generate_simulated_vlm_outputs(
                batch_size=B, 
                device=device,
                max_seq_len=None
            )
        else:
            vl_feat = nobs['vqa'].to(dtype=torch.float32)
        
        nsample, denoising_steps = self.conditional_sample_with_steps(
            cond_data, 
            cond_mask,
            cond=cond,
            vl_features=vl_feat
        )
        
        naction_pred = nsample[...,:Da]
        naction_pred = torch.clamp(naction_pred, -1.0, 1.0)
        
        # 反归一化去噪步骤用于可视化
        denoising_steps_unnormalized = []
        if self.enable_action_normalization and self.action_stats is not None:
            for step in denoising_steps:
                step_actions = step[..., :Da]
                step_actions_clamped = torch.clamp(step_actions, -1.0, 1.0)
                step_actions_unnorm = self.unnormalize_action(step_actions_clamped)
                denoising_steps_unnormalized.append(step_actions_unnorm)
            
            # 反归一化最终预测
            naction_pred = self.unnormalize_action(naction_pred)
            device = naction_pred.device
            action_min = self.action_stats['min'].to(device)
            action_max = self.action_stats['max'].to(device)
            safety_margin = 0.2
            range_size = action_max - action_min
            expanded_min = action_min - safety_margin * range_size
            expanded_max = action_max + safety_margin * range_size
            naction_pred = torch.clamp(naction_pred, expanded_min, expanded_max)
        else:
            denoising_steps_unnormalized = [step[..., :Da] for step in denoising_steps]
        
        action_pred = naction_pred.detach().cpu().numpy()
        action = action_pred
        result = {
            'action': action,
            'action_pred': action_pred
        }
        
        # 转换去噪步骤为numpy（已经反归一化）
        denoising_steps_np = [step[0].detach().cpu().numpy() for step in denoising_steps_unnormalized]
        
        return result, denoising_steps_np

    def conditional_sample_with_steps(self, 
            condition_data, condition_mask,
            cond=None, generator=None, vl_features=None,
            **kwargs):
        """
        条件采样并返回中间去噪步骤
        """
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        scheduler.set_timesteps(self.num_inference_steps)
        
        # Use provided vl_features or generate simulated ones
        if vl_features is None:
            print("No VLM features provided, generating simulated features...")
            vl_features, vl_mask = self.generate_simulated_vlm_outputs(trajectory.shape[0], trajectory.device)
        else:
            # vl_features provided, create mask for valid positions
            vl_mask = torch.ones(vl_features.shape[:2], dtype=torch.bool, device=vl_features.device)
        
        vl_embeds = self.feature_encoder(vl_features)

        # 保存去噪步骤
        denoising_steps = []
        denoising_steps.append(trajectory.clone())  # 初始噪声
        
        total_steps = len(scheduler.timesteps)
        for i, t in enumerate(scheduler.timesteps):
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, vl_embeds, cond, vl_mask=vl_mask)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
            
            # 保存关键步骤（初始、33%、66%、最终）
            if i == total_steps // 3 or i == 2 * total_steps // 3:
                denoising_steps.append(trajectory.clone())
        
        # 最终结果
        trajectory[condition_mask] = condition_data[condition_mask]
        denoising_steps.append(trajectory.clone())

        return trajectory, denoising_steps

    def load_lidar_bev_encoder(self):
        # Load BEV encoder configuration from config file
        bev_encoder_cfg = self.config.get('bev_encoder', {})
        obs_encoder = InterfuserBEVEncoder(
            perception_backbone=None,
            state_dim=bev_encoder_cfg.get('state_dim', 10),
            feature_dim=bev_encoder_cfg.get('feature_dim', 256),
            use_group_norm=bev_encoder_cfg.get('use_group_norm', True),
            freeze_backbone=bev_encoder_cfg.get('freeze_backbone', False),
            bev_input_size=tuple(bev_encoder_cfg.get('bev_input_size', [448, 448]))
        )
        
        # Load pretrained weights from config
        pretrained_path = bev_encoder_cfg.get('pretrained_path', None)
        if pretrained_path is not None and os.path.exists(pretrained_path):
            load_lidar_submodules(obs_encoder, pretrained_path, strict=False, logger=None)
            print(f"✓ BEV encoder loaded from: {pretrained_path}")
        else:
            print(f"⚠ BEV encoder pretrained_path not found or not specified: {pretrained_path}")
            print("  Continuing with random initialization...")
        return obs_encoder

    def load_vit_model(self):
        """
        Load the VIT model with pretrained weights
        
        Args:
            model_path: Path to the pretrained model directory
            device: Device to load the model on
            
        Returns:
            vit_model: Loaded VIT model in eval mode
            vit_transform: Image transformation function
            vit_config: VIT configuration
        """
        vit_model_path = self.config.get('vit_model_path', None)
        image_transform_args = self.config.get('image_transform_args', {})
        print(f"Loading VIT model from: {vit_model_path}")
        
        # Initialize tokenizer (required for add_special_tokens)
        tokenizer = Qwen2Tokenizer.from_pretrained(vit_model_path)
        tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
        
        # Load VIT config
        vit_config = SiglipVisionConfig.from_json_file(os.path.join(vit_model_path, "vit_config.json"))
        vit_select_layer = -2 
        vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + vit_select_layer
        vit_config.rope = True
        
        # Create VIT model
        vit_model = SiglipVisionModel(vit_config)
        vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
        
        # Create image transform
        vit_transform = ImageTransform(**image_transform_args)
        
        # Move to device and set to eval mode
        vit_model = vit_model.to(self.device)
        vit_model.eval()
        
        # Freeze all parameters
        for param in vit_model.parameters():
            param.requires_grad = False
        
        print("✓ VIT model loaded successfully")
        
        return vit_model, vit_transform, vit_config
    
    def preprocess_rgb_image(image_path, vit_transform):
        """
        Load and preprocess an RGB image
        
        Args:
            image_path: Path to the RGB image
            vit_transform: Image transformation function
            
        Returns:
            image_tensor: Preprocessed image tensor
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = pil_img2rgb(image)
        
        # Apply VIT transform
        image_tensor = vit_transform(image, img_num=1)
        
        return image_tensor
    # ========= VLM feature simulati, temporary! TODO   ============

    def _init_loaded_vlm_features(self):
        """
        从预先保存的文件中加载VLM特征
        """
        print("Loading VLM features from file...")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        template_path = os.path.join(project_root, 'fixed_vlm_template.pt')
        if os.path.exists(template_path):
            self.fixed_vlm_template = torch.load(template_path)
            self.fixed_seq_len = self.fixed_vlm_template.shape[0]
            print(f"✓ VLM features loaded: shape={self.fixed_vlm_template.shape}, seq_len={self.fixed_seq_len}")
            # 根据实际VLM特征维度创建特征编码器
            vlm_feature_dim = self.fixed_vlm_template.shape[1]  # 隐藏层维度
            self.feature_encoder = nn.Linear(vlm_feature_dim, 1536)
            self.feature_encoder.eval()
            print(f"✓ Feature encoder created: {vlm_feature_dim} -> 1536")
        else:
            print("⚠ VLM feature file not found, using simulated features")
            self._init_fixed_vlm_features()

    def _init_fixed_vlm_features(self):
        """
        初始化真实的VLM特征,从实际的VLM模型中提取隐藏层特征
        如果VLM模型不可用,则使用固定的随机特征作为备用
        """
        print("Initializing VLM features...")
        
        # 检查VLM backbone是否可用
        if self.vlm_backbone is not None:
            try:
                print("Attempting to use real VLM model...")
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                # 创建一个简单的图像输入来获取VLM特征
                # 使用PIL创建一个测试图像，这是Qwen VL模型所期望的格式
                import numpy as np
                from PIL import Image
                
                # 创建一个固定的测试图像 (224x224 RGB)
                test_image_array = np.random.RandomState(42).randint(0, 255, (224, 224, 3), dtype=np.uint8)
                test_image = Image.fromarray(test_image_array)
                test_text = "What actions should be taken based on this scene?"
                
                try:
                    # 使用VLM的processor来正确处理图像和文本
                    messages = [
                        {
                            "role": "user", 
                            "content": [
                                {"type": "image", "image": test_image},
                                {"type": "text", "text": test_text}
                            ]
                        }
                    ]
                    
                    # 应用聊天模板
                    text_inputs = self.vlm_backbone.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    
                    # 处理输入
                    inputs = self.vlm_backbone.tokenizer(
                        text=[text_inputs],
                        images=[test_image], 
                        return_tensors="pt",
                        padding=True
                    )
                    
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    if device == 'cuda' and hasattr(self.vlm_backbone, 'model'):
                        self.vlm_backbone.model = self.vlm_backbone.model.to(device)
                    
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # 获取VLM模型的隐藏状态
                    with torch.no_grad():
                        outputs = self.vlm_backbone.model(
                            **inputs,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        
                        # 获取最后一层隐藏状态
                        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                            hidden_states = outputs.hidden_states[-1]  
                            
                            # 移除batch维度并转换为float32
                            self.fixed_vlm_template = hidden_states.squeeze(0).float()  # (seq_len, hidden_size)
                            self.fixed_seq_len = self.fixed_vlm_template.shape[0]
                            
                            print(f"✓ Real VLM features initialized: shape={self.fixed_vlm_template.shape}, seq_len={self.fixed_seq_len}")
                            
                            # 根据实际VLM特征维度创建特征编码器
                            vlm_feature_dim = self.fixed_vlm_template.shape[1]  # 隐藏层维度
                            self.feature_encoder = nn.Linear(vlm_feature_dim, 1536)
                            self.feature_encoder.eval()
                            print(f"✓ Feature encoder created: {vlm_feature_dim} -> 1536")
                            self.fixed_vlm_template = self.fixed_vlm_template.cpu()
                            
                            if device == 'cuda':
                                self.vlm_backbone.model = self.vlm_backbone.model.cpu()
                                del self.vlm_backbone.model
                                self.vlm_backbone.model = None
                                #torch.cuda.empty_cache()
                                print("✓ VLM model moved to CPU and GPU memory cleared")
                            
                            return
                        else:
                            print("⚠ VLM model output does not contain hidden_states, falling back to simulated features")
                            
                except Exception as inner_e:
                    print(f"⚠ Error in VLM processing: {inner_e}")
                    
            except Exception as e:
                print(f"⚠ Failed to initialize real VLM features: {e}")
        
        # 备用方案：使用固定的随机特征
        print("Using simulated VLM features...")
        F = 2560  # VLM隐藏层维度
        self.fixed_seq_len = 8  # 固定序列长度
        
        generator = torch.Generator()
        generator.manual_seed(42)  
        
        # 生成单个固定的VLM特征模板 (seq_len, F)
        self.fixed_vlm_template = torch.randn(
            self.fixed_seq_len, F, 
            generator=generator, 
            dtype=torch.float32
        )
        
        print(f"✓ Simulated VLM features initialized: shape={self.fixed_vlm_template.shape}, seq_len={self.fixed_seq_len}")
        
        # 根据VLM特征维度创建特征编码器
        if self.feature_encoder is None:
            vlm_feature_dim = self.fixed_vlm_template.shape[1]  # 隐藏层维度 
            self.feature_encoder = nn.Linear(vlm_feature_dim, 1536)
            self.feature_encoder.eval()
            print(f"✓ Feature encoder created: {vlm_feature_dim} -> 1536")
    
    def generate_simulated_vlm_outputs(self, batch_size, device, max_seq_len=None):
        """
        生成真实的VLM输出特征, 支持可变序列长度和padding
        
        Args:
            batch_size: 批次大小 (B)
            device: 设备 ('cuda' 或 'cpu')
            max_seq_len: 最大序列长度, 用于padding (可选)
            
        Returns:
            vlm_features: 形状为 (B, seq_len, F) 的张量
            vl_mask: 形状为 (B, seq_len) 的bool张量,True表示有效位置, False表示padding
        """
        # 将固定模板移动到指定设备
        template_on_device = self.fixed_vlm_template.to(device)
        current_seq_len = template_on_device.shape[0]
        
        # 如果指定了最大序列长度且当前序列较短，则进行padding
        if max_seq_len is not None and current_seq_len < max_seq_len:
            # 创建padding
            padding_size = max_seq_len - current_seq_len
            feature_dim = template_on_device.shape[1]
            padding = torch.zeros(padding_size, feature_dim, device=device, dtype=template_on_device.dtype)
            
            # 添加padding到模板
            padded_template = torch.cat([template_on_device, padding], dim=0)
            
            # 创建mask：True为有效位置，False为padding位置
            vl_mask = torch.ones(max_seq_len, dtype=torch.bool, device=device)
            vl_mask[current_seq_len:] = False
            
            seq_len = max_seq_len
            template_to_use = padded_template
        else:
            # 不需要padding，所有位置都是有效的
            vl_mask = torch.ones(current_seq_len, dtype=torch.bool, device=device)
            seq_len = current_seq_len
            template_to_use = template_on_device
        
        # 为批次重复
        if batch_size == 1:
            vlm_features = template_to_use.unsqueeze(0)  # (1, seq_len, F)
            vl_mask_batch = vl_mask.unsqueeze(0)  # (1, seq_len)
        else:
            vlm_features = template_to_use.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, seq_len, F)
            vl_mask_batch = vl_mask.unsqueeze(0).repeat(batch_size, 1)  # (B, seq_len)
        
        return vlm_features, vl_mask_batch



    # ===============Copy from TCP=====================


    def control_pid(self, waypoints, velocity, target):
        ''' Predicts vehicle control with a PID controller.
		Args:
			waypoints (tensor): output of self.plan()
			velocity (tensor): speedometer input
		'''
        # Read hyperparameters from config
        control_cfg = self.config.get('controller', {})
        aim_dist = control_cfg.get('aim_dist', 4.0)  # distance to search around for aim point
        angle_thresh = control_cfg.get('angle_thresh', 0.3)  # outlier control detection angle
        dist_thresh = control_cfg.get('dist_thresh', 10.0)  # target point y-distance for outlier filtering
        brake_speed = control_cfg.get('brake_speed', 0.4)  # desired speed below which brake is triggered
        brake_ratio = control_cfg.get('brake_ratio', 1.1)  # ratio of speed to desired speed at which brake is triggered
        clip_delta = control_cfg.get('clip_delta', 0.25)  # maximum change in speed input to longitudinal controller
        max_throttle = control_cfg.get('max_throttle', 0.75)  # upper limit on throttle signal value in dataset


        assert(waypoints.size(0)==1)
        waypoints = waypoints[0].data.cpu().numpy()
        target = target.squeeze().data.cpu().numpy()
        
        waypoints[:, [0, 1]] = waypoints[:, [1, 0]]  
        target[[0, 1]] = target[[1, 0]]

		# Downsample waypoints: from 10Hz (20 points in 2s) to 2Hz (take every 5th point)
		# Original indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
		# Downsampled indices: 4, 9, 14, 19 (starting from index 4)
		# This matches the original 2Hz assumption: 4 waypoints with 2.5s intervals = 10 seconds
        downsample_factor = 5
        downsampled_waypoints = waypoints[4::downsample_factor]

		# iterate over vectors between predicted waypoints
        num_pairs = len(downsampled_waypoints) - 1
        best_norm = 1e5
        desired_speed = 0
        aim = downsampled_waypoints[0]
        for i in range(num_pairs):
            # magnitude of vectors, used for speed
            desired_speed += np.linalg.norm(
					downsampled_waypoints[i+1] - downsampled_waypoints[i]) * 2.0 / num_pairs
            # norm of vector midpoints, used for steering
            norm = np.linalg.norm((downsampled_waypoints[i+1] + downsampled_waypoints[i]) / 2.0)
            if abs(aim_dist-best_norm) > abs(aim_dist-norm):
                aim = downsampled_waypoints[i]
                best_norm = norm
        
        aim_last = downsampled_waypoints[-1] - downsampled_waypoints[-2]
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
        angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

		# choice of point to aim for steering, removing outlier predictions
		# use target point if it has a smaller angle or if error is large
		# predicted point otherwise
		# (reduces noise in eg. straight roads, helps with sudden turn commands)
        use_target_to_aim = np.abs(angle_target) < np.abs(angle)
        use_target_to_aim = use_target_to_aim or (np.abs(angle_target-angle_last) > angle_thresh and target[1] < dist_thresh)
        if use_target_to_aim:
            angle_final = angle_target
        else:
            angle_final = angle
        
        steer = self.turn_controller.step(angle_final)
        steer = np.clip(steer, -1.0, 1.0)

        speed = velocity[0].data.cpu().numpy()
        brake = desired_speed < brake_speed or (speed / desired_speed) > brake_ratio

        delta = np.clip(desired_speed - speed, 0.0, clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, max_throttle)
        throttle = throttle if not brake else 0.0

        metadata = {
			'speed': float(speed.astype(np.float64)),
			'steer': float(steer),
			'throttle': float(throttle),
			'brake': float(brake),
			'wp_4': tuple(downsampled_waypoints[3].astype(np.float64)) if len(downsampled_waypoints) > 3 else tuple(downsampled_waypoints[-1].astype(np.float64)),
			'wp_3': tuple(downsampled_waypoints[2].astype(np.float64)) if len(downsampled_waypoints) > 2 else tuple(downsampled_waypoints[-1].astype(np.float64)),
			'wp_2': tuple(downsampled_waypoints[1].astype(np.float64)) if len(downsampled_waypoints) > 1 else tuple(downsampled_waypoints[-1].astype(np.float64)),
			'wp_1': tuple(downsampled_waypoints[0].astype(np.float64)),
			'aim': tuple(aim.astype(np.float64)),
			'target': tuple(target.astype(np.float64)),
			'desired_speed': float(desired_speed.astype(np.float64)),
			'angle': float(angle.astype(np.float64)),
			'angle_last': float(angle_last.astype(np.float64)),
			'angle_target': float(angle_target.astype(np.float64)),
			'angle_final': float(angle_final.astype(np.float64)),
			'delta': float(delta.astype(np.float64)),
		}

        return steer, throttle, brake, metadata