"""Official LEAD NAVSIM image preprocessing helpers.

These utilities mirror LEAD's vendored NAVSIM v1.1 TransfuserFeatureBuilder
and CarlaTransfuserAgent path:

1. stitch cam_l0, cam_f0, cam_r0, cam_b0 horizontally,
2. resize the full stitched image by 1/4,
3. JPEG encode with quality 30,
4. decode with OpenCV IMREAD_COLOR and feed the resulting BGR tensor.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import torch

from navsim.common.dataclasses import AgentInput, SensorConfig


LEAD_NAVSIM_CAMERA_ATTRS = ("cam_l0", "cam_f0", "cam_r0", "cam_b0")
LEAD_NAVSIM_RAW_CAMERA_KEYS = ("CAM_L0", "CAM_F0", "CAM_R0", "CAM_B0")
LEAD_NAVSIM_JPEG_QUALITY = 30


def build_official_lead_sensor_config(include_history: bool = False) -> SensorConfig:
    """Request the four NAVSIM cameras needed by official LEAD preprocessing."""

    frames = [0, 1, 2, 3] if include_history else [3]
    return SensorConfig(
        cam_f0=frames,
        cam_l0=frames,
        cam_r0=frames,
        cam_b0=frames,
        cam_l1=False,
        cam_l2=False,
        cam_r1=False,
        cam_r2=False,
        lidar_pc=False,
    )


def _as_uint8_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HWC 3-channel image, got shape={arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def stitch_official_lead_navsim_4cam(images: Sequence[np.ndarray]) -> np.ndarray:
    """Stitch L0/F0/R0/B0 exactly as LEAD's NAVSIM feature builder does."""

    if len(images) != len(LEAD_NAVSIM_CAMERA_ATTRS):
        raise ValueError(f"Expected 4 camera images, got {len(images)}")
    return np.concatenate([_as_uint8_image(img) for img in images], axis=1)


def encode_official_lead_camera_feature_from_arrays(images: Sequence[np.ndarray]) -> np.ndarray:
    """Return the compressed camera_feature bytes used by official LEAD NAVSIM."""

    stitched = stitch_official_lead_navsim_4cam(images)
    resized = cv2.resize(stitched, (stitched.shape[1] // 4, stitched.shape[0] // 4))
    ok, compressed = cv2.imencode(
        ".jpg",
        resized,
        [int(cv2.IMWRITE_JPEG_QUALITY), LEAD_NAVSIM_JPEG_QUALITY],
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode LEAD NAVSIM camera_feature")
    return compressed


def encode_official_lead_camera_feature(agent_input: AgentInput) -> np.ndarray:
    """Build official LEAD compressed camera_feature bytes from an AgentInput."""

    cameras = agent_input.cameras[-1]
    images = [getattr(cameras, name).image for name in LEAD_NAVSIM_CAMERA_ATTRS]
    return encode_official_lead_camera_feature_from_arrays(images)


def decode_official_lead_camera_feature(camera_feature: np.ndarray) -> np.ndarray:
    """Decode official LEAD camera_feature bytes to the BGR image tensor input."""

    compressed = np.frombuffer(camera_feature, dtype=np.uint8)
    decoded = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("OpenCV failed to decode LEAD NAVSIM camera_feature")
    return decoded


def build_official_lead_rgb_tensor_from_arrays(
    images: Sequence[np.ndarray],
    *,
    batched: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the NCHW/CHW tensor fed to the LEAD model."""

    camera_feature = encode_official_lead_camera_feature_from_arrays(images)
    decoded = decode_official_lead_camera_feature(camera_feature)
    tensor = torch.from_numpy(decoded.copy()).permute(2, 0, 1).to(dtype=dtype)
    if batched:
        tensor = tensor.unsqueeze(0)
    return tensor


def build_official_lead_rgb_tensor(
    agent_input: AgentInput,
    *,
    batched: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build official LEAD NAVSIM camera tensor from an AgentInput."""

    camera_feature = encode_official_lead_camera_feature(agent_input)
    decoded = decode_official_lead_camera_feature(camera_feature)
    tensor = torch.from_numpy(decoded.copy()).permute(2, 0, 1).to(dtype=dtype)
    if batched:
        tensor = tensor.unsqueeze(0)
    return tensor
