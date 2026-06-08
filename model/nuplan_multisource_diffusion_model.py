"""N2-style multi-source diffusion model for nuPlan.

The model keeps the existing nuPlan vector SceneEncoder and ego+neighbor
trajectory contract. It replaces the simple DiT decoder with a source-aware
decoder inspired by the N2 unified skeleton.

Two route concepts are intentionally separate:
- ``route_lanes`` are external road-structure context from the PlanTF cache.
- ``sampled_route`` is the optional ego route-geometry diffusion target.
"""

import torch
import torch.nn as nn
from timm.models.layers import Mlp

from model.adapter_layer import AdapterLayer
from model.dit import FinalLayer, TimestepEmbedder, modulate
from model.lora_linear import LoRALinear
from model.nuplan_diffusion_model import ModuleAttrMixin, _cfg_attr
from model.scene_encoder import SceneEncoder


class RouteLaneContextEncoder(nn.Module):
    """Encode on-route lane geometry as external source tokens."""

    def __init__(
        self,
        route_num: int = 10,
        lane_len: int = 20,
        hidden_dim: int = 192,
        point_hidden_dim: int = 128,
    ):
        super().__init__()
        self.lane_len = int(lane_len)
        self.hidden_dim = int(hidden_dim)
        self.point_mlp = Mlp(
            in_features=4,
            hidden_features=point_hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.center_pos_emb = nn.Linear(4, hidden_dim)
        self.type_emb = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.global_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, route_lanes: torch.Tensor):
        """
        Args:
            route_lanes: (B, route_num, lane_len, 4), [x, y, dx, dy]

        Returns:
            context_tokens: (B, route_num, D)
            context_mask:   (B, route_num), True for padded route lanes
            context_global: (B, D)
        """
        B, R, L, _ = route_lanes.shape
        valid_points = torch.sum(torch.ne(route_lanes[..., :4], 0), dim=-1) != 0
        context_mask = torch.sum(valid_points, dim=-1) == 0

        point_tokens = self.point_mlp(route_lanes.reshape(B * R * L, 4))
        point_tokens = point_tokens.reshape(B, R, L, self.hidden_dim)
        point_tokens = point_tokens * valid_points.unsqueeze(-1).to(point_tokens.dtype)

        denom = valid_points.sum(dim=-1, keepdim=True).clamp_min(1).to(point_tokens.dtype)
        context_tokens = point_tokens.sum(dim=2) / denom

        center_idx = min(self.lane_len // 2, L - 1)
        context_tokens = context_tokens + self.center_pos_emb(route_lanes[:, :, center_idx, :4])
        context_tokens = self.norm(context_tokens + self.type_emb)
        context_tokens = context_tokens.masked_fill(context_mask.unsqueeze(-1), 0.0)

        valid_routes = (~context_mask).unsqueeze(-1).to(context_tokens.dtype)
        global_denom = valid_routes.sum(dim=1).clamp_min(1.0)
        context_global = (context_tokens * valid_routes).sum(dim=1) / global_denom
        context_global = self.global_proj(context_global)
        return context_tokens, context_mask, context_global


class MultiSourceAttention(nn.Module):
    """Single-softmax attention over self, scene, and road-context sources."""

    def __init__(self, dim: int = 192, heads: int = 6, dropout: float = 0.1):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim must be divisible by heads, got dim={dim}, heads={heads}")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.scale = self.head_dim ** -0.5

        self.q_self = nn.Linear(dim, dim)
        self.q_scene_adapter = nn.Linear(dim, dim)
        self.q_road_adapter = nn.Linear(dim, dim)
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)
        self.k_scene = nn.Linear(dim, dim)
        self.v_scene = nn.Linear(dim, dim)
        self.k_road = nn.Linear(dim, dim)
        self.v_road = nn.Linear(dim, dim)

        self.source_bias = nn.Parameter(torch.zeros(3, heads))
        self.source_logit_scale = nn.Parameter(torch.zeros(3, heads))
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.reshape(B, L, self.heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _mask_scores(scores: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        if key_mask is None:
            return scores
        min_value = torch.finfo(scores.dtype).min
        return scores.masked_fill(key_mask[:, None, None, :], min_value)

    def forward(
        self,
        x: torch.Tensor,
        scene_tokens: torch.Tensor,
        road_tokens: torch.Tensor,
        self_key_mask: torch.Tensor = None,
        scene_key_mask: torch.Tensor = None,
        road_key_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        q_base = self._shape(self.q_self(x))
        q_scene = q_base + self._shape(self.q_scene_adapter(x))
        q_road = q_base + self._shape(self.q_road_adapter(x))

        k_self = self._shape(self.k_self(x))
        v_self = self._shape(self.v_self(x))
        k_scene = self._shape(self.k_scene(scene_tokens))
        v_scene = self._shape(self.v_scene(scene_tokens))
        k_road = self._shape(self.k_road(road_tokens))
        v_road = self._shape(self.v_road(road_tokens))

        source_scale = torch.exp(self.source_logit_scale).view(3, 1, self.heads, 1, 1)
        source_bias = self.source_bias.view(3, 1, self.heads, 1, 1)

        attn_self = (q_base @ k_self.transpose(-2, -1)) * self.scale
        attn_scene = (q_scene @ k_scene.transpose(-2, -1)) * self.scale
        attn_road = (q_road @ k_road.transpose(-2, -1)) * self.scale
        attn_self = attn_self * source_scale[0] + source_bias[0]
        attn_scene = attn_scene * source_scale[1] + source_bias[1]
        attn_road = attn_road * source_scale[2] + source_bias[2]

        attn_self = self._mask_scores(attn_self, self_key_mask)
        attn_scene = self._mask_scores(attn_scene, scene_key_mask)
        attn_road = self._mask_scores(attn_road, road_key_mask)

        attn = torch.cat([attn_self, attn_scene, attn_road], dim=-1)
        weights = self.dropout(torch.softmax(attn, dim=-1))
        values = torch.cat([v_self, v_scene, v_road], dim=2)
        out = weights @ values
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.dim)
        return self.out_proj(out)


class NuPlanMultiSourceBlock(nn.Module):
    """AdaLN decoder block with N2-style multi-source attention."""

    def __init__(
        self,
        dim: int = 192,
        heads: int = 6,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.multi_attn = MultiSourceAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=hidden,
            out_features=dim,
            act_layer=approx_gelu,
            drop=0.0,
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(
        self,
        x: torch.Tensor,
        scene_tokens: torch.Tensor,
        road_tokens: torch.Tensor,
        y: torch.Tensor,
        self_key_mask: torch.Tensor,
        scene_key_mask: torch.Tensor,
        road_key_mask: torch.Tensor,
    ) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(y).chunk(6, dim=1)
        )

        attn_in = modulate(self.norm1(x), shift_attn, scale_attn)
        x = x + gate_attn.unsqueeze(1) * self.multi_attn(
            attn_in,
            scene_tokens,
            road_tokens,
            self_key_mask=self_key_mask,
            scene_key_mask=scene_key_mask,
            road_key_mask=road_key_mask,
        )
        adapter_attn = getattr(self, "adapter_attn", None)
        if adapter_attn is not None:
            x = adapter_attn(x)

        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)
        adapter_mlp = getattr(self, "adapter_mlp", None)
        if adapter_mlp is not None:
            x = adapter_mlp(x)
        return x


class NuPlanMultiSourceDecoder(nn.Module):
    """Joint decoder for trajectory tokens and optional ego route tokens."""

    def __init__(
        self,
        hidden_dim: int = 192,
        heads: int = 6,
        depth: int = 3,
        traj_output_dim: int = 324,
        route_points: int = 50,
        predict_route_tokens: bool = False,
        route_num: int = 10,
        lane_len: int = 20,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.traj_output_dim = int(traj_output_dim)
        self.route_points = int(route_points)
        self.predict_route_tokens = bool(predict_route_tokens)

        preproj_hidden = max(512, hidden_dim * 2)
        self.traj_preproj = Mlp(
            in_features=traj_output_dim,
            hidden_features=preproj_hidden,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.agent_embedding = nn.Embedding(2, hidden_dim)
        if self.predict_route_tokens:
            self.route_preproj = Mlp(
                in_features=2,
                hidden_features=max(128, hidden_dim // 2),
                out_features=hidden_dim,
                act_layer=nn.GELU,
                drop=0.0,
            )
            self.route_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.route_pos_embedding = nn.Parameter(torch.zeros(1, self.route_points, hidden_dim))
            nn.init.normal_(self.route_pos_embedding, std=0.02)

        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.road_context_encoder = RouteLaneContextEncoder(
            route_num=route_num,
            lane_len=lane_len,
            hidden_dim=hidden_dim,
            point_hidden_dim=max(128, hidden_dim // 2),
        )
        self.blocks = nn.ModuleList(
            [
                NuPlanMultiSourceBlock(hidden_dim, heads, dropout, mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.traj_final_layer = FinalLayer(hidden_dim, traj_output_dim)
        if self.predict_route_tokens:
            self.route_final_layer = FinalLayer(hidden_dim, 2)

    def _embed_trajectory_tokens(self, sampled_trajectories: torch.Tensor) -> torch.Tensor:
        B, P, _ = sampled_trajectories.shape
        x = self.traj_preproj(sampled_trajectories)
        agent_emb = torch.cat(
            [
                self.agent_embedding.weight[0][None, :],
                self.agent_embedding.weight[1][None, :].expand(P - 1, -1),
            ],
            dim=0,
        )
        return x + agent_emb[None]

    def forward(
        self,
        sampled_trajectories: torch.Tensor,
        timestep: torch.Tensor,
        scene_tokens: torch.Tensor,
        scene_mask: torch.Tensor,
        route_lanes: torch.Tensor,
        neighbor_current_mask: torch.Tensor,
        sampled_route: torch.Tensor = None,
    ) -> dict:
        B, P, _ = sampled_trajectories.shape
        traj_tokens = self._embed_trajectory_tokens(sampled_trajectories)
        tokens = traj_tokens

        if self.predict_route_tokens:
            if sampled_route is None:
                raise KeyError("sampled_route is required when predict_route_tokens=True")
            if sampled_route.shape[1] != self.route_points or sampled_route.shape[-1] != 2:
                raise ValueError(
                    f"Expected sampled_route shape (B,{self.route_points},2), "
                    f"got {tuple(sampled_route.shape)}"
                )
            route_tokens = self.route_preproj(sampled_route)
            route_tokens = route_tokens + self.route_type_embedding + self.route_pos_embedding
            tokens = torch.cat([tokens, route_tokens], dim=1)

        road_tokens, road_mask, road_global = self.road_context_encoder(route_lanes)
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0).expand(B)
        y = self.t_embedder(timestep) + road_global

        self_mask = torch.zeros((B, tokens.shape[1]), dtype=torch.bool, device=tokens.device)
        self_mask[:, 1:P] = neighbor_current_mask

        for block in self.blocks:
            tokens = block(
                tokens,
                scene_tokens,
                road_tokens,
                y,
                self_key_mask=self_mask,
                scene_key_mask=scene_mask,
                road_key_mask=road_mask,
            )

        traj_out = tokens[:, :P]
        result = {"traj": self.traj_final_layer(traj_out, y)}
        if self.predict_route_tokens:
            route_out = tokens[:, P:]
            result["route"] = self.route_final_layer(route_out, y)
        return result


class NuPlanMultiSourceDiffusionModel(ModuleAttrMixin):
    """NuPlan white-noise diffusion with an N2-style multi-source decoder."""

    def __init__(self, config):
        super().__init__()
        c = config
        self.hidden_dim = _cfg_attr(c, "hidden_dim", 192)
        self.future_len = _cfg_attr(c, "future_len", 80)
        self.predicted_neighbor_num = _cfg_attr(c, "predicted_neighbor_num", 10)
        self.P = 1 + self.predicted_neighbor_num
        self.output_dim = (self.future_len + 1) * 4
        self.route_points = int(_cfg_attr(c, "route_points", _cfg_attr(c, "route_token_points", 50)))
        self.predict_route_tokens = bool(
            _cfg_attr(c, "predict_route_tokens", _cfg_attr(c, "n2_predict_route_tokens", False))
        )

        self.scene_encoder = SceneEncoder(config)
        self.decoder = NuPlanMultiSourceDecoder(
            hidden_dim=self.hidden_dim,
            heads=_cfg_attr(c, "num_heads", 6),
            depth=_cfg_attr(c, "decoder_depth", 3),
            traj_output_dim=self.output_dim,
            route_points=self.route_points,
            predict_route_tokens=self.predict_route_tokens,
            route_num=_cfg_attr(c, "route_num", 10),
            lane_len=_cfg_attr(c, "lane_len", 20),
            dropout=_cfg_attr(c, "decoder_drop_path_rate", 0.1),
            mlp_ratio=_cfg_attr(c, "decoder_mlp_ratio", 4.0),
        )

        self._peft_config = _cfg_attr(c, "peft_config", None) or {}
        self.peft_active = False
        if self._peft_config:
            self._inject_peft(self._peft_config)

    def _inject_peft(self, peft_config):
        peft_type = str(peft_config.get("type", "adapter")).lower()
        rank = int(peft_config.get("rank", 8))
        alpha = float(peft_config.get("alpha", rank if peft_type == "lora" else 1.0))
        if peft_type not in {"adapter", "lora"}:
            raise ValueError(
                f"peft_config.type must be 'adapter' or 'lora', got '{peft_type}'"
            )
        if rank <= 0:
            raise ValueError(f"peft_config.rank must be > 0, got {rank}")

        for param in self.parameters():
            param.requires_grad_(False)
        if peft_type == "adapter":
            self._inject_adapter(rank)
        else:
            self._inject_lora(rank, alpha)
        self.peft_active = True

    def _inject_adapter(self, rank: int) -> None:
        for block in self.decoder.blocks:
            block.add_module("adapter_attn", AdapterLayer(self.hidden_dim, rank))
            block.add_module("adapter_mlp", AdapterLayer(self.hidden_dim, rank))

    def _inject_lora(self, rank: int, alpha: float) -> None:
        replaced = self._replace_linear_with_lora(self.decoder, rank, alpha)
        if replaced == 0:
            raise RuntimeError("LoRA requested but no nn.Linear modules were found under decoder")

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
        B = inputs["neighbor_agents_past"].shape[0]
        encoder_outputs = self.scene_encoder(inputs)

        neighbor_current_mask = inputs.get("neighbor_current_mask")
        if neighbor_current_mask is None:
            neighbor_current = inputs["neighbor_agents_past"][
                :, :self.predicted_neighbor_num, -1, :4
            ]
            neighbor_current_mask = torch.sum(torch.ne(neighbor_current, 0), dim=-1) == 0

        decoded = self.decoder(
            sampled_trajectories=inputs["sampled_trajectories"],
            timestep=inputs["diffusion_time"],
            scene_tokens=encoder_outputs["encoding"],
            scene_mask=encoder_outputs["mask"],
            route_lanes=inputs["route_lanes"],
            neighbor_current_mask=neighbor_current_mask,
            sampled_route=inputs.get("sampled_route"),
        )
        score = decoded["traj"].reshape(B, self.P, self.future_len + 1, 4)
        result = {"score": score}
        if "route" in decoded:
            result["route_score"] = decoded["route"]
        return result
