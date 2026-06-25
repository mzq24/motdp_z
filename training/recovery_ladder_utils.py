"""Utilities for deterministic legacy-e60 recovery experiments."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import torch


SEMANTIC_NAME_MARKERS = (
    "shared_stage1_",
    "semantic_transition_",
    "semantic_prev_",
    "traj_window_condition_proj",
    "traj_dir_condition_proj",
    "traj_decision_phase_condition_proj",
    "traj_control_phase_condition_proj",
    "traj_boundary_margin_proj",
    "traj_opportunity_condition_proj",
    "traj_area_status_condition_proj",
    "traj_timing_condition_proj",
    "traj_chase_condition_proj",
    "traj_current_edge_condition_proj",
    "traj_future_edge_condition_proj",
    "traj_edge_margin_proj",
    "traj_edge_valid_proj",
    "traj_borrow_aux_proj",
    "route_prev_coarse_memory_proj",
)


def set_global_seed(seed: int) -> None:
    """Seed model initialization and dataloader base-seed generation."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_semantic_parameter(name: str) -> bool:
    return any(marker in name for marker in SEMANTIC_NAME_MARKERS)


def freeze_unused_semantic_modules(model: torch.nn.Module) -> Tuple[int, int]:
    """Freeze semantic-only parameters while leaving the motion path trainable."""
    frozen_params = 0
    frozen_tensors = 0
    for name, parameter in model.named_parameters():
        if is_semantic_parameter(name):
            parameter.requires_grad_(False)
            frozen_params += parameter.numel()
            frozen_tensors += 1
    return frozen_params, frozen_tensors


def trainable_parameters(model: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def _checkpoint_state(checkpoint: Mapping) -> Mapping[str, torch.Tensor]:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_motion_common(
    model: torch.nn.Module,
    checkpoint_path: str,
) -> Dict[str, object]:
    """Load shape-compatible non-semantic parameters from an initialization checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = _checkpoint_state(checkpoint)
    target = model.state_dict()
    matched = {}
    skipped_semantic = []
    skipped_shape = []
    unexpected = []
    for name, tensor in source.items():
        if is_semantic_parameter(name):
            skipped_semantic.append(name)
        elif name not in target:
            unexpected.append(name)
        elif tuple(target[name].shape) != tuple(tensor.shape):
            skipped_shape.append(name)
        else:
            matched[name] = tensor
    missing, _ = model.load_state_dict(matched, strict=False)
    missing_motion = [name for name in missing if not is_semantic_parameter(name)]
    if missing_motion:
        preview = missing_motion[:8]
        raise RuntimeError(
            "motion_common initialization did not cover all motion parameters: "
            f"{preview}{'...' if len(missing_motion) > 8 else ''}"
        )
    return {
        "checkpoint": str(checkpoint_path),
        "matched_tensor_count": len(matched),
        "skipped_semantic_count": len(skipped_semantic),
        "skipped_shape_count": len(skipped_shape),
        "unexpected_count": len(unexpected),
    }


def apply_recovery_initialization(model: torch.nn.Module, config: Dict) -> Dict[str, object]:
    training_cfg = config.get("training", {})
    checkpoint_path = training_cfg.get("init_checkpoint")
    if not checkpoint_path:
        return {"scope": "random", "checkpoint": None}
    scope = str(training_cfg.get("init_scope", "motion_common"))
    if scope != "motion_common":
        raise ValueError(f"Unsupported recovery init_scope={scope!r}")
    report = load_motion_common(model, checkpoint_path)
    report["scope"] = scope
    return report


def _tensor_hash(named_tensors: Iterable[Tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def motion_initialization_hash(model: torch.nn.Module) -> str:
    return _tensor_hash(
        (name, tensor)
        for name, tensor in model.state_dict().items()
        if not is_semantic_parameter(name)
    )


def _git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
        ).strip()
    except Exception:
        return "unknown"


def write_structure_manifest(
    *,
    model: torch.nn.Module,
    config: Dict,
    config_path: str,
    checkpoint_dir: Path,
    policy_class: str,
    effective_lr: float,
    init_report: Dict[str, object],
    project_root: Path,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_bytes = Path(config_path).read_bytes()
    named_parameters = list(model.named_parameters())
    semantic_parameters = [
        (name, parameter) for name, parameter in named_parameters if is_semantic_parameter(name)
    ]
    route_cfg = config.get("route_b", {})
    manifest = {
        "git_commit": _git_commit(project_root),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "policy_class": policy_class,
        "model_class": type(getattr(model, "model", model)).__name__,
        "parameter_count": sum(parameter.numel() for _, parameter in named_parameters),
        "trainable_parameter_count": sum(
            parameter.numel() for _, parameter in named_parameters if parameter.requires_grad
        ),
        "semantic_parameter_count": sum(parameter.numel() for _, parameter in semantic_parameters),
        "semantic_trainable_parameter_count": sum(
            parameter.numel() for _, parameter in semantic_parameters if parameter.requires_grad
        ),
        "motion_initialization_sha256": motion_initialization_hash(model),
        "configured_lr": float(config.get("optimizer", {}).get("lr", 0.0)),
        "effective_lr": float(effective_lr),
        "seed": int(config.get("training", {}).get("seed", 0)),
        "initialization": init_report,
        "flags": {
            key: route_cfg.get(key)
            for key in (
                "motion_only_model",
                "motion_objective",
                "train_stage1",
                "use_stage1_state",
                "use_traj_branch_condition",
                "use_semantic_state_transition",
                "use_cover_relation_graph_decoder",
                "use_route_prev_coarse_memory",
                "use_route_intent_token",
                "use_lidar_bev_detail",
                "freeze_unused_semantic_modules",
            )
        },
    }
    path = checkpoint_dir / "structure_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def save_initial_checkpoint(model: torch.nn.Module, checkpoint_dir: Path, config: Dict) -> Path:
    path = checkpoint_dir / "initial_model.pt"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 0, "config": config}, path)
    return path
