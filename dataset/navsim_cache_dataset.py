"""Feature-only NAVSIM TransFuser cache reader for LEAD/MoT-DP smoke tests.

This module intentionally stops before label conversion. NAVSIM target files are
indexed only so we can verify paired cache completeness; trajectory/route/stage1
labels should be mapped in a separate step.
"""

from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


PathLike = Union[str, Path]
TensorDict = Dict[str, torch.Tensor]

LEAD_INPUT_RANKS = {
    "rgb": 3,
    "status_feature": 1,
    "command": 1,
    "speed": 0,
    "acceleration": 0,
}


@dataclass(frozen=True)
class NavsimCacheEntry:
    feature_path: Path
    target_path: Optional[Path]
    log_name: str
    cache_token: str


def load_gzip_pickle(path: PathLike) -> Any:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def load_navsim_cache_feature(path: PathLike) -> Mapping[str, Any]:
    feature = load_gzip_pickle(path)
    if not isinstance(feature, Mapping):
        raise TypeError(f"Expected mapping in feature cache {path}, got {type(feature)!r}")
    return feature


def load_navsim_cache_target(path: PathLike) -> Mapping[str, Any]:
    target = load_gzip_pickle(path)
    if not isinstance(target, Mapping):
        raise TypeError(f"Expected mapping in target cache {path}, got {type(target)!r}")
    return target


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _encoded_u8_array(value: Any) -> np.ndarray:
    if isinstance(value, bytes):
        return np.frombuffer(value, dtype=np.uint8)
    array = _to_numpy(value)
    if array.dtype != np.uint8:
        array = array.astype(np.uint8, copy=False)
    return array.reshape(-1)


def decode_camera_feature(camera_feature: Any) -> np.ndarray:
    """Decode NAVSIM compressed camera feature into LEAD-compatible CHW image."""

    array = _to_numpy(camera_feature)
    if array.ndim == 3:
        image = array
    else:
        encoded = _encoded_u8_array(camera_feature)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to decode camera_feature")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected decoded HWC image with 3 channels, got {image.shape}")
    return np.transpose(image, (2, 0, 1)).copy()


def feature_to_lead_input(feature: Mapping[str, Any]) -> TensorDict:
    """Build the non-label LEAD input dict from one NAVSIM cached feature."""

    if "camera_feature" not in feature:
        raise KeyError("NAVSIM cache feature is missing 'camera_feature'")
    if "status_feature" not in feature:
        raise KeyError("NAVSIM cache feature is missing 'status_feature'")

    rgb = torch.from_numpy(decode_camera_feature(feature["camera_feature"]))
    status = torch.as_tensor(feature["status_feature"], dtype=torch.float32)
    if status.ndim != 1 or status.shape[0] < 8:
        raise ValueError(f"Expected status_feature shape [>=8], got {tuple(status.shape)}")

    return {
        "rgb": rgb,
        "status_feature": status,
        "command": status[:4],
        "speed": torch.linalg.norm(status[4:6]),
        "acceleration": torch.linalg.norm(status[6:8]),
    }


def find_navsim_cache_entries(
    cache_root: PathLike,
    require_target: bool = True,
    max_samples: Optional[int] = None,
) -> List[NavsimCacheEntry]:
    root = Path(cache_root).expanduser()
    feature_paths = sorted(root.glob("**/transfuser_feature.gz"))
    entries: List[NavsimCacheEntry] = []
    for feature_path in feature_paths:
        target_path = feature_path.with_name("transfuser_target.gz")
        has_target = target_path.exists()
        if require_target and not has_target:
            continue

        entries.append(
            NavsimCacheEntry(
                feature_path=feature_path,
                target_path=target_path if has_target else None,
                log_name=feature_path.parent.parent.name,
                cache_token=feature_path.parent.name,
            )
        )
        if max_samples is not None and len(entries) >= max_samples:
            break

    if not entries:
        target_note = " with paired transfuser_target.gz" if require_target else ""
        raise RuntimeError(f"No transfuser_feature.gz files{target_note} found under {root}")
    return entries


class NavsimTransfuserCacheDataset(Dataset):
    """Read NAVSIM TransFuser cache features without mapping training labels."""

    def __init__(
        self,
        cache_root: PathLike,
        require_target: bool = True,
        max_samples: Optional[int] = None,
        load_raw_feature: bool = False,
        load_raw_target: bool = False,
    ) -> None:
        self.cache_root = Path(cache_root).expanduser()
        self.entries = find_navsim_cache_entries(
            self.cache_root,
            require_target=require_target,
            max_samples=max_samples,
        )
        self.load_raw_feature = bool(load_raw_feature)
        self.load_raw_target = bool(load_raw_target)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.entries[index]
        feature = load_navsim_cache_feature(entry.feature_path)
        sample: Dict[str, Any] = feature_to_lead_input(feature)
        sample.update(
            {
                "feature_path": str(entry.feature_path),
                "target_path": str(entry.target_path) if entry.target_path is not None else "",
                "has_target": entry.target_path is not None,
                "log_name": entry.log_name,
                "cache_token": entry.cache_token,
                "index": index,
            }
        )

        if self.load_raw_feature:
            sample["navsim_feature_raw"] = feature
        if self.load_raw_target:
            if entry.target_path is None:
                raise RuntimeError(f"No paired target for {entry.feature_path}")
            sample["navsim_target_raw"] = load_navsim_cache_target(entry.target_path)
        return sample


def ensure_lead_batch(
    sample: Mapping[str, Any],
    device: Optional[Union[str, torch.device]] = None,
    keys: Sequence[str] = tuple(LEAD_INPUT_RANKS.keys()),
) -> TensorDict:
    """Add batch dims to a single sample, or pass through an already collated batch."""

    batch: TensorDict = {}
    torch_device = torch.device(device) if device is not None else None
    for key in keys:
        if key not in sample:
            raise KeyError(f"Missing LEAD input key: {key}")
        value = sample[key]
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        expected_rank = LEAD_INPUT_RANKS[key]
        if tensor.ndim == expected_rank:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != expected_rank + 1:
            raise ValueError(
                f"Expected {key} rank {expected_rank} or {expected_rank + 1}, "
                f"got shape {tuple(tensor.shape)}"
            )
        if torch_device is not None:
            tensor = tensor.to(torch_device)
        batch[key] = tensor
    return batch


def make_zero_lidar_bev_batch(
    batch_size: int,
    history_frames: int = 1,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a non-label LiDAR BEV placeholder for MoT-DP observation smoke tests."""

    return torch.zeros(
        int(batch_size),
        int(history_frames),
        2,
        256,
        256,
        device=device,
        dtype=dtype,
    )
