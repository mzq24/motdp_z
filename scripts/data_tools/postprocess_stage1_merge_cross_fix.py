#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np


MERGE_DECISION_PHASE_TO_CODE = {
    "none": 0,
    "yld": 1,
    "go": 2,
}

MERGE_END_STATE_TO_CODE = {
    "none": 0,
    "ended_with_chase": 1,
    "ended_with_cross": 2,
    "ended_with_other_current_actor": 3,
    "ended_empty": 4,
    "ended_route_end": 5,
}


def _atomic_pickle_dump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + f".tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def _sample_current_speed_mps(sample: dict) -> float:
    speed_hist = np.asarray(sample.get("speed_hist"), dtype=np.float32).reshape(-1)
    if speed_hist.size > 0:
        value = float(speed_hist[-1])
        if np.isfinite(value):
            return value
    ego_status = np.asarray(sample.get("ego_status"), dtype=np.float32)
    if ego_status.ndim >= 2 and ego_status.shape[-1] >= 1:
        value = float(ego_status.reshape(-1, ego_status.shape[-1])[-1, 0])
        if np.isfinite(value):
            return value
    return 0.0


def _cover_interaction_name(cover):
    return str(((cover or {}).get("interaction") or {}).get("name", "none"))


def _cover_interaction_subtype(cover):
    interaction = ((cover or {}).get("interaction") or {})
    return str(interaction.get("subtype") or interaction.get("name") or "none")


def _cover_is_cross_meet(cover):
    if int((cover or {}).get("exists", 0.0)) <= 0:
        return False
    if _cover_interaction_name(cover) != "meet":
        return False
    return "cross" in _cover_interaction_subtype(cover)


def _cover_start_distance_m(cover):
    if not _cover_is_cross_meet(cover):
        return np.nan
    d = (cover or {}).get("distance", (cover or {}).get("d_ego", np.nan))
    try:
        d = float(d)
    except Exception:
        return np.nan
    return d if np.isfinite(d) else np.nan


def _cross_start_distance_m(sample):
    stage1_debug = sample.get("stage1_speed_debug")
    if not isinstance(stage1_debug, dict):
        return np.nan

    speed_curve = stage1_debug.get("speed_curve", {})
    if not isinstance(speed_curve, dict):
        speed_curve = {}
    meet_debug = speed_curve.get("meet_debug", {})
    if not isinstance(meet_debug, dict):
        meet_debug = {}

    subtype = str(meet_debug.get("subtype", "none"))
    if subtype == "borrow_cross_meet":
        borrow_start = float(meet_debug.get("borrow_start_distance_m", np.nan))
        if np.isfinite(borrow_start):
            return borrow_start

    distances = []
    for cover_key in ("current_cover", "speed_curve_future_cover"):
        d = _cover_start_distance_m(stage1_debug.get(cover_key) or {})
        if np.isfinite(d):
            distances.append(float(d))

    meet_d_ego = float(meet_debug.get("d_ego_m", np.nan))
    if np.isfinite(meet_d_ego) and subtype != "borrow_cross_meet" and "cross" in subtype:
        distances.append(meet_d_ego)

    if not distances:
        return np.nan
    return float(min(distances))


def _base_dir_from_sample(sample):
    feat = sample.get("transfuser_bev_feature", "")
    return os.path.dirname(os.path.dirname(feat)) if feat else ""


def _frame_id_from_sample(sample):
    return int(sample.get("frame_id", -1))


def _group_indices_by_scene(samples):
    scene_to_indices = {}
    scene_order = []
    for idx, sample in enumerate(samples):
        base_dir = _base_dir_from_sample(sample)
        if base_dir not in scene_to_indices:
            scene_to_indices[base_dir] = []
            scene_order.append(base_dir)
        scene_to_indices[base_dir].append(idx)
    ordered = []
    for base_dir in scene_order:
        indices = scene_to_indices[base_dir]
        indices.sort(key=lambda idx: _frame_id_from_sample(samples[idx]))
        ordered.append((base_dir, indices))
    return ordered


def _ensure_stage1_debug(sample: dict) -> dict:
    stage1_debug = sample.get("stage1_speed_debug")
    if not isinstance(stage1_debug, dict):
        stage1_debug = {}
        sample["stage1_speed_debug"] = stage1_debug
    return stage1_debug


def _ensure_debug_speed_curve(sample):
    stage1_debug = _ensure_stage1_debug(sample)
    speed_curve = stage1_debug.get("speed_curve")
    if not isinstance(speed_curve, dict):
        speed_curve = {}
        stage1_debug["speed_curve"] = speed_curve
    return speed_curve


def _default_merge_episode_debug() -> dict:
    return {
        "phase": "none",
        "phase_code": int(MERGE_DECISION_PHASE_TO_CODE["none"]),
        "episode_id": -1,
        "active": 0.0,
        "no_go": 0.0,
        "start_frame": -1,
        "end_frame": -1,
        "go_frame": -1,
        "resolution_actor_id": -1,
        "end_state": "none",
        "end_state_code": int(MERGE_END_STATE_TO_CODE["none"]),
    }


def _set_stage1_merge_defaults(sample: dict) -> None:
    sample["merge_decision_phase"] = np.int64(MERGE_DECISION_PHASE_TO_CODE["none"])
    sample["merge_episode_id"] = np.int64(-1)
    sample["merge_episode_active"] = np.float32(0.0)
    sample["merge_episode_no_go"] = np.float32(0.0)
    sample["merge_episode_start_frame"] = np.int64(-1)
    sample["merge_episode_end_frame"] = np.int64(-1)
    sample["merge_go_frame"] = np.int64(-1)
    sample["merge_resolution_actor_id"] = np.int64(-1)
    sample["merge_end_state"] = np.int64(MERGE_END_STATE_TO_CODE["none"])
    stage1_debug = _ensure_stage1_debug(sample)
    stage1_debug["merge_episode"] = dict(_default_merge_episode_debug())


def _set_stage1_merge_annotation(sample: dict, merge_info: dict) -> None:
    phase = str(merge_info.get("phase", "none"))
    end_state = str(merge_info.get("end_state", "none"))
    sample["merge_decision_phase"] = np.int64(MERGE_DECISION_PHASE_TO_CODE.get(phase, 0))
    sample["merge_episode_id"] = np.int64(int(merge_info.get("episode_id", -1)))
    sample["merge_episode_active"] = np.float32(float(merge_info.get("active", 0.0)))
    sample["merge_episode_no_go"] = np.float32(float(merge_info.get("no_go", 0.0)))
    sample["merge_episode_start_frame"] = np.int64(int(merge_info.get("start_frame", -1)))
    sample["merge_episode_end_frame"] = np.int64(int(merge_info.get("end_frame", -1)))
    sample["merge_go_frame"] = np.int64(int(merge_info.get("go_frame", -1)))
    sample["merge_resolution_actor_id"] = np.int64(int(merge_info.get("resolution_actor_id", -1)))
    sample["merge_end_state"] = np.int64(MERGE_END_STATE_TO_CODE.get(end_state, 0))
    stage1_debug = _ensure_stage1_debug(sample)
    stage1_debug["merge_episode"] = dict(merge_info)


def _find_last_contiguous_true_block(values: List[bool]) -> Tuple[int, int] | None:
    if not values or not values[-1]:
        return None
    end = len(values) - 1
    start = end
    while start > 0 and values[start - 1]:
        start -= 1
    return start, end


def _find_first_consecutive_run(values: List[bool], run_len: int) -> Tuple[int, int] | None:
    if run_len <= 0:
        return None
    count = 0
    run_start = None
    for idx, flag in enumerate(values):
        if flag:
            if count == 0:
                run_start = idx
            count += 1
            if count >= run_len and run_start is not None:
                return run_start, idx
        else:
            count = 0
            run_start = None
    return None


def _find_contiguous_true_runs(values: List[bool]) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(values):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _fix_merge_scene(
    samples: List[dict],
    indices: List[int],
    speed_thresh: float,
    consecutive_frames: int,
    stop_speed_thresh: float,
    post_go_fallback_frames: int,
) -> bool:
    active_vals = [float(samples[idx].get("merge_episode_active", 0.0)) > 0.5 for idx in indices]
    tail_block = _find_last_contiguous_true_block(active_vals)
    if tail_block is None:
        return False

    block_start_local, block_end_local = tail_block
    block_indices = indices[block_start_local : block_end_local + 1]
    go_frame_value = max(int(samples[idx].get("merge_go_frame", -1)) for idx in block_indices)
    has_go_frame = go_frame_value >= 0
    no_go_scene = any(float(samples[idx].get("merge_episode_no_go", 0.0)) > 0.5 for idx in block_indices)
    if has_go_frame:
        go_local = None
        for local_pos, sample_idx in enumerate(block_indices):
            if int(samples[sample_idx].get("frame_id", -1)) == int(go_frame_value):
                go_local = local_pos
                break
        if go_local is None:
            return False
        fallback_end_local = min(int(go_local) + int(post_go_fallback_frames), len(block_indices) - 1)
        if fallback_end_local >= len(block_indices) - 1:
            return False

        changed = False
        for local_pos, sample_idx in enumerate(block_indices):
            if local_pos <= fallback_end_local:
                continue
            _set_stage1_merge_defaults(samples[sample_idx])
            changed = True
        if not changed:
            return False

        new_end_idx = block_indices[fallback_end_local]
        new_end_frame = _frame_id_from_sample(samples[new_end_idx])
        new_end_state = "ended_route_end" if new_end_idx == indices[-1] else "ended_with_chase"
        for local_pos, sample_idx in enumerate(block_indices[: fallback_end_local + 1]):
            phase = "yld" if local_pos < go_local else "go"
            info = dict((_ensure_stage1_debug(samples[sample_idx]).get("merge_episode") or {}))
            info.update({
                "phase": phase,
                "phase_code": int(MERGE_DECISION_PHASE_TO_CODE[phase]),
                "active": 1.0,
                "no_go": 0.0,
                "go_frame": int(go_frame_value),
                "end_frame": int(new_end_frame),
                "end_state": str(new_end_state),
                "end_state_code": int(MERGE_END_STATE_TO_CODE.get(new_end_state, 0)),
                "postprocess_post_go_fallback_frames": int(post_go_fallback_frames),
            })
            _set_stage1_merge_annotation(samples[sample_idx], info)
        return True

    if not no_go_scene:
        return False

    speeds = [_sample_current_speed_mps(samples[idx]) for idx in block_indices]
    moving = [float(v) > float(speed_thresh) for v in speeds]
    run = _find_first_consecutive_run(moving, int(consecutive_frames))
    if run is None:
        return False

    run_start_local, run_end_local = run
    if run_start_local <= 0:
        return False

    go_local = None
    for pos in range(run_start_local - 1, -1, -1):
        if float(speeds[pos]) <= float(stop_speed_thresh):
            go_local = pos
            break
    if go_local is None:
        return False

    start_idx = block_indices[0]
    end_idx = block_indices[run_end_local]
    go_idx = block_indices[go_local]
    episode_id = int(samples[start_idx].get("merge_episode_id", -1))
    resolution_actor_id = int(samples[start_idx].get("merge_resolution_actor_id", -1))
    end_state_code = int(samples[end_idx].get("merge_end_state", MERGE_END_STATE_TO_CODE["none"]))
    end_state = next(
        (name for name, code in MERGE_END_STATE_TO_CODE.items() if int(code) == int(end_state_code)),
        "none",
    )
    if end_state == "none" and end_idx == indices[-1]:
        end_state = "ended_route_end"

    start_frame = _frame_id_from_sample(samples[start_idx])
    end_frame = _frame_id_from_sample(samples[end_idx])
    go_frame = _frame_id_from_sample(samples[go_idx])

    for local_pos, sample_idx in enumerate(block_indices):
        if local_pos > run_end_local:
            _set_stage1_merge_defaults(samples[sample_idx])
            continue
        phase = "yld" if local_pos < go_local else "go"
        merge_info = {
            "phase": phase,
            "phase_code": int(MERGE_DECISION_PHASE_TO_CODE[phase]),
            "episode_id": int(episode_id),
            "active": 1.0,
            "no_go": 0.0,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "go_frame": int(go_frame),
            "resolution_actor_id": int(resolution_actor_id),
            "end_state": str(end_state),
            "end_state_code": int(MERGE_END_STATE_TO_CODE.get(end_state, 0)),
            "frame_role": "fallback_go" if local_pos >= go_local else "fallback_yld",
            "fallback_inferred_go": 1.0,
            "fallback_speed_thresh_mps": float(speed_thresh),
            "fallback_consecutive_frames": int(consecutive_frames),
            "fallback_stop_speed_thresh_mps": float(stop_speed_thresh),
        }
        _set_stage1_merge_annotation(samples[sample_idx], merge_info)
    return True


def _recompute_cross_scene(
    samples: List[dict],
    indices: List[int],
    cross_start_distance_m: float,
    cross_wait_speed_thresh: float,
    cross_go_speed_thresh: float,
    cross_wait_dt_s: float,
) -> int:
    records = []
    for idx in indices:
        sample = samples[idx]
        stage1_debug = sample.get("stage1_speed_debug")
        current_cover = stage1_debug.get("current_cover", {}) if isinstance(stage1_debug, dict) else {}
        future_cover = stage1_debug.get("speed_curve_future_cover", {}) if isinstance(stage1_debug, dict) else {}
        raw_cross_active = _cover_is_cross_meet(current_cover) or _cover_is_cross_meet(future_cover)
        d_start = _cross_start_distance_m(sample)
        near_cross_start = np.isfinite(d_start) and float(d_start) <= float(cross_start_distance_m)
        ego_speed = float(_sample_current_speed_mps(sample))
        wait_now = bool(raw_cross_active and near_cross_start and ego_speed <= float(cross_wait_speed_thresh))
        go_now = bool(raw_cross_active and near_cross_start and ego_speed >= float(cross_go_speed_thresh))
        candidate_now = bool(raw_cross_active and near_cross_start)
        records.append({
            "sample_idx": int(idx),
            "raw_cross_active": bool(raw_cross_active),
            "near_cross_start": bool(near_cross_start),
            "wait_now": bool(wait_now),
            "go_now": bool(go_now),
            "candidate_now": bool(candidate_now),
        })

    candidate_runs = _find_contiguous_true_runs([record["candidate_now"] for record in records])
    cross_active_count = 0
    keep_run_mask = [False] * len(records)
    cross_wait_time_values = [0.0] * len(records)
    cross_wait_valid_values = [0.0] * len(records)
    cross_go_values = [0.0] * len(records)

    for run_start, run_end in candidate_runs:
        run_wait_any = any(records[pos]["wait_now"] for pos in range(run_start, run_end + 1))
        run_go_any = any(records[pos]["go_now"] for pos in range(run_start, run_end + 1))
        if not (run_wait_any or run_go_any):
            continue
        for pos in range(run_start, run_end + 1):
            keep_run_mask[pos] = True
        wait_frames = 0
        for pos in range(run_start, run_end + 1):
            if records[pos]["wait_now"]:
                wait_frames += 1
            else:
                wait_frames = 0
            cross_wait_time_values[pos] = float(wait_frames) * float(cross_wait_dt_s)
            cross_wait_valid_values[pos] = 1.0 if records[pos]["wait_now"] else 0.0
            cross_go_values[pos] = 1.0 if records[pos]["go_now"] else 0.0

    for pos, record in enumerate(records):
        sample = samples[record["sample_idx"]]
        cross_active = 1.0 if keep_run_mask[pos] else 0.0
        sample["cross_active"] = float(cross_active)
        sample["cross_episode_active"] = float(cross_active)
        sample["speed_cross_wait_time_s"] = np.float32(cross_wait_time_values[pos])
        sample["speed_cross_wait_valid"] = np.float32(cross_wait_valid_values[pos])
        if cross_active > 0.5:
            cross_active_count += 1

        speed_curve = _ensure_debug_speed_curve(sample)
        speed_curve["cross_active"] = float(cross_active)
        speed_curve["cross_episode_active"] = float(cross_active)
        speed_curve["cross_wait_time_s"] = float(cross_wait_time_values[pos])
        speed_curve["cross_wait_valid"] = float(cross_wait_valid_values[pos])
        speed_curve["cross_wait_time_s_recomputed"] = float(cross_wait_time_values[pos])
        speed_curve["cross_wait_valid_recomputed"] = float(cross_wait_valid_values[pos])
        speed_curve["cross_active_raw_recomputed"] = 1.0 if record["raw_cross_active"] else 0.0
        speed_curve["cross_active_near_start_recomputed"] = 1.0 if record["near_cross_start"] else 0.0
        speed_curve["cross_go_recomputed"] = float(cross_go_values[pos])

    return cross_active_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix stage1 merge no-go tails and recompute cross_active on an existing packed file.")
    parser.add_argument("--input_path", required=True, help="Existing samples_packed.pkl")
    parser.add_argument("--output_path", required=True, help="Output pickle path; may equal input_path")
    parser.add_argument("--merge_speed_thresh", type=float, default=2.0)
    parser.add_argument("--merge_consecutive_frames", type=int, default=5)
    parser.add_argument("--merge_stop_speed_thresh", type=float, default=0.5)
    parser.add_argument("--merge_post_go_fallback_frames", type=int, default=6)
    parser.add_argument("--cross_start_distance_m", type=float, default=10.0)
    parser.add_argument("--cross_wait_speed_thresh", type=float, default=0.5)
    parser.add_argument("--cross_go_speed_thresh", type=float, default=1.0)
    parser.add_argument("--cross_wait_dt_s", type=float, default=0.25)
    args = parser.parse_args()

    with open(args.input_path, "rb") as f:
        samples = pickle.load(f)

    merge_fixed_scenes = 0
    cross_active_samples = 0
    scene_count = 0
    for _, indices in _group_indices_by_scene(samples):
        scene_count += 1
        if _fix_merge_scene(
            samples,
            indices,
            speed_thresh=float(args.merge_speed_thresh),
            consecutive_frames=int(args.merge_consecutive_frames),
            stop_speed_thresh=float(args.merge_stop_speed_thresh),
            post_go_fallback_frames=int(args.merge_post_go_fallback_frames),
        ):
            merge_fixed_scenes += 1
        cross_active_samples += _recompute_cross_scene(
            samples,
            indices,
            cross_start_distance_m=float(args.cross_start_distance_m),
            cross_wait_speed_thresh=float(args.cross_wait_speed_thresh),
            cross_go_speed_thresh=float(args.cross_go_speed_thresh),
            cross_wait_dt_s=float(args.cross_wait_dt_s),
        )

    _atomic_pickle_dump(samples, args.output_path)

    print(
        {
            "scenes": scene_count,
            "samples": len(samples),
            "merge_fixed_scenes": merge_fixed_scenes,
            "cross_active_samples": cross_active_samples,
            "output": args.output_path,
        }
    )


if __name__ == "__main__":
    main()
