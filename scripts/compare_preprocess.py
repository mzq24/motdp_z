"""Compare MoT-DP LEAD preprocessing against official LEAD NAVSIM v1.1."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
LEAD_NAVSIM_ROOT = Path("/workspace1/z_project/code/lead/3rd_party/navsim_workspace/navsimv1.1")
for path in (REPO_ROOT, LEAD_NAVSIM_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader

from navsim_motdp.lead_preprocessing import (
    build_official_lead_rgb_tensor,
    build_official_lead_sensor_config,
    decode_official_lead_camera_feature,
    encode_official_lead_camera_feature,
)


def main() -> None:
    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        frame_interval=1,
        has_route=True,
        max_scenes=1,
    )
    sensor_config = build_official_lead_sensor_config()

    loader = SceneLoader(
        data_path=Path("/workspace2/data/navsim/navsim_logs/mini"),
        sensor_blobs_path=Path("/workspace2/data/navsim/sensor_blobs/mini"),
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )

    token = loader.tokens[0]
    agent_input = loader.get_agent_input_from_token(token)
    print(f"Token: {token}")

    feature_builder = TransfuserFeatureBuilder(TransfuserConfig())
    official_feature = feature_builder._get_camera_feature(agent_input)
    official_bytes = official_feature.detach().cpu().numpy()
    motdp_bytes = encode_official_lead_camera_feature(agent_input)

    official_decoded = decode_official_lead_camera_feature(official_bytes)
    motdp_decoded = decode_official_lead_camera_feature(motdp_bytes)
    motdp_tensor = build_official_lead_rgb_tensor(agent_input, batched=True)
    official_tensor = torch.from_numpy(official_decoded.copy()).permute(2, 0, 1).unsqueeze(0).float()

    print()
    print("=== Official LEAD v1.1 TransfuserFeatureBuilder ===")
    print(f"camera_feature bytes: shape={official_bytes.shape}, dtype={official_bytes.dtype}, size={official_bytes.nbytes}")
    print(f"decoded: shape={official_decoded.shape}, dtype={official_decoded.dtype}")

    print()
    print("=== MoT-DP helper ===")
    print(f"camera_feature bytes: shape={motdp_bytes.shape}, dtype={motdp_bytes.dtype}, size={motdp_bytes.nbytes}")
    print(f"decoded: shape={motdp_decoded.shape}, dtype={motdp_decoded.dtype}")
    print(f"tensor: shape={tuple(motdp_tensor.shape)}, dtype={motdp_tensor.dtype}")

    byte_equal = np.array_equal(official_bytes, motdp_bytes)
    decoded_abs = np.abs(official_decoded.astype(np.int16) - motdp_decoded.astype(np.int16))
    tensor_abs = (official_tensor - motdp_tensor).abs()

    print()
    print("=== Parity ===")
    print(f"compressed bytes equal: {byte_equal}")
    print(f"decoded max_abs={decoded_abs.max()}, mean_abs={decoded_abs.mean():.6f}")
    print(f"tensor max_abs={tensor_abs.max().item():.6f}, mean_abs={tensor_abs.mean().item():.6f}")

    assert byte_equal, "MoT-DP compressed camera_feature differs from official LEAD"
    assert tuple(motdp_tensor.shape) == (1, 3, 270, 1920)
    assert torch.equal(official_tensor, motdp_tensor), "MoT-DP decoded tensor differs from official LEAD"
    print()
    print("PASS: MoT-DP preprocessing is byte/tensor identical to official LEAD NAVSIM v1.1.")


if __name__ == "__main__":
    main()
