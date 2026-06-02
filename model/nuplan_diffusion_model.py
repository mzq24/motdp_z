"""NuPlan Diffusion Model — combined SceneEncoder + DiT.

Jointly predicts ego + neighbor future trajectories from white noise,
conditioned on ego-centric scene context (agents, map, route, static objects).

Architecture:
    Input (ego-centric vectors) -> SceneEncoder -> scene tokens (B, N, 192)
    Noisy trajectories (B, P, output_dim) -> DiT -> predicted x0 trajectories

Config can be a dict or an object with attribute access.
"""

import logging
import torch
import torch.nn as nn

from model.adapter_layer import AdapterLayer
from model.scene_encoder import SceneEncoder
from model.lora_linear import LoRALinear
from model.dit import DiT


logger = logging.getLogger(__name__)


def _cfg_attr(config, key, default=None):
    """Get config value from either dict or namespace."""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class ModuleAttrMixin(nn.Module):
    """Mixin that makes config attributes accessible via __getattr__."""

    def __init__(self):
        super().__init__()
        self._config_initialized = False

    def update_attr(self, name: str, value):
        if '.' in name:
            parts = name.split('.')
            obj = self
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            setattr(self, name, value)


class NuPlanDiffusionModel(ModuleAttrMixin):
    """White-noise diffusion model for nuPlan trajectory prediction.

    Encodes ego-centric scene context via vector-based encoders,
    then denoises joint ego+neighbor trajectories via a DiT decoder.

    Inputs:
        neighbor_agents_past:  (B, 32, 21, 11)  agent history
        static_objects:        (B, 5, 10)        static objects
        lanes:                 (B, 30, 20, 12)   vector map lanes
        lanes_speed_limit:     (B, 30, 1)        speed limit per lane
        lanes_has_speed_limit: (B, 30, 1)        speed limit valid flag
        route_lanes:           (B, 10, 20, 4)    route lane geometry
        sampled_trajectories:  (B, P, output_dim) flattened noisy trajectories
        diffusion_time:        (B,)               diffusion timestep
        neighbor_current_mask: (B, P-1)           True = padded neighbor

    Output:
        score: (B, P, future_len+1, 4)  predicted clean trajectories
    """

    def __init__(self, config):
        super().__init__()

        c = config
        self.hidden_dim = _cfg_attr(c, 'hidden_dim', 192)
        self.future_len = _cfg_attr(c, 'future_len', 80)
        self.predicted_neighbor_num = _cfg_attr(c, 'predicted_neighbor_num', 10)
        self.P = 1 + self.predicted_neighbor_num
        self.output_dim = (self.future_len + 1) * 4

        self.scene_encoder = SceneEncoder(config)
        self.dit = DiT(
            hidden_dim=self.hidden_dim,
            heads=_cfg_attr(c, 'num_heads', 6),
            depth=_cfg_attr(c, 'decoder_depth', 3),
            output_dim=self.output_dim,
            route_num=_cfg_attr(c, 'route_num', 10),
            lane_len=_cfg_attr(c, 'lane_len', 20),
            dropout=_cfg_attr(c, 'decoder_drop_path_rate', 0.1),
        )

        self._peft_config = _cfg_attr(c, 'peft_config', None) or {}
        self.peft_active = False
        if self._peft_config:
            self._inject_peft(self._peft_config)

    def _inject_peft(self, peft_config):
        peft_type = str(peft_config.get('type', 'adapter')).lower()
        rank = int(peft_config.get('rank', 8))
        alpha = float(peft_config.get('alpha', rank if peft_type == 'lora' else 1.0))

        if peft_type not in {'adapter', 'lora'}:
            raise ValueError(
                f"peft_config.type must be 'adapter' or 'lora', got '{peft_type}'"
            )
        if rank <= 0:
            raise ValueError(f"peft_config.rank must be > 0, got {rank}")

        for param in self.parameters():
            param.requires_grad_(False)

        if peft_type == 'adapter':
            self._inject_adapter(rank)
        else:
            self._inject_lora(rank, alpha)

        self.peft_active = True

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        logger.info(
            "PEFT injection done (type=%s, rank=%d, alpha=%.4g): trainable %d / %d params (%.1f%%)",
            peft_type,
            rank,
            alpha,
            n_trainable,
            n_total,
            100.0 * n_trainable / max(n_total, 1),
        )

    def _inject_adapter(self, rank: int) -> None:
        for block in self.dit.blocks:
            block.add_module('adapter_self_attn', AdapterLayer(self.hidden_dim, rank))
            block.add_module('adapter_mlp1', AdapterLayer(self.hidden_dim, rank))
            block.add_module('adapter_cross_attn', AdapterLayer(self.hidden_dim, rank))
            block.add_module('adapter_mlp2', AdapterLayer(self.hidden_dim, rank))

    def _inject_lora(self, rank: int, alpha: float) -> None:
        replaced = self._replace_linear_with_lora(self.dit, rank, alpha)
        if replaced == 0:
            raise RuntimeError('LoRA PEFT requested but no nn.Linear modules were found under self.dit')
        logger.info(
            "LoRA replaced %d nn.Linear modules under DiT decoder (rank=%d, alpha=%.4g)",
            replaced,
            rank,
            alpha,
        )

    def _replace_linear_with_lora(self, module: nn.Module, rank: int, alpha: float) -> int:
        if isinstance(module, nn.MultiheadAttention):
            return 0
        replaced = 0
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear.from_linear(child, rank, alpha))
                replaced += 1
            else:
                replaced += self._replace_linear_with_lora(child, rank, alpha)
        return replaced

    def forward(self, inputs: dict) -> dict:
        B = inputs['neighbor_agents_past'].shape[0]

        encoder_outputs = self.scene_encoder(inputs)
        scene_encoding = encoder_outputs['encoding']

        if 'neighbor_current_mask' in inputs:
            neighbor_current_mask = inputs['neighbor_current_mask']
        else:
            neighbor_current = inputs['neighbor_agents_past'][
                :, :self.predicted_neighbor_num, -1, :4
            ]
            neighbor_current_mask = (
                torch.sum(torch.ne(neighbor_current, 0), dim=-1) == 0
            )

        x = inputs['sampled_trajectories']
        t = inputs['diffusion_time']
        route_lanes = inputs['route_lanes']

        predicted = self.dit(
            x=x, t=t,
            scene_encoding=scene_encoding,
            route_lanes=route_lanes,
            neighbor_current_mask=neighbor_current_mask,
        )

        score = predicted.reshape(B, self.P, self.future_len + 1, 4)
        return {'score': score}
