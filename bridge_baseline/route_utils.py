"""
Utility: convert (route, target_speed) → time-spaced trajectory
using pure-pursuit geometry + bicycle kinematics.

NOT used in training or evaluation yet.
See docs/bridge_baseline_v2.md §Route-to-Traj for rationale.
"""

import numpy as np


def route_speed_to_traj(
    route: np.ndarray,
    target_speed: float,
    dt: float = 0.5,
    n_steps: int = 6,
    wheelbase: float = 2.7,
    max_steer_angle: float = 0.6,   # radians
) -> np.ndarray:
    """
    Simulate vehicle motion along `route` at constant `target_speed`
    using pure-pursuit steering + bicycle kinematics.

    Args:
        route:        (N, 2) ego-relative waypoints in metres, x=forward, y=left
        target_speed: scalar m/s (model's predicted target speed)
        dt:           time step in seconds (default 0.5s)
        n_steps:      number of output waypoints (default 6 → 3s horizon)
        wheelbase:    vehicle wheelbase in metres
        max_steer_angle: maximum steering angle in radians

    Returns:
        traj: (n_steps, 2) ego-relative trajectory waypoints
    """
    route = np.asarray(route, dtype=np.float64)   # (N, 2)
    v = float(target_speed)

    # State: ego frame, starts at origin facing +x
    x, y, yaw = 0.0, 0.0, 0.0
    traj = []

    # Pre-compute cumulative arc lengths along route for interpolation
    deltas = np.diff(route, axis=0)                        # (N-1, 2)
    seg_lens = np.linalg.norm(deltas, axis=1)              # (N-1,)
    cum_lens = np.concatenate([[0.0], np.cumsum(seg_lens)])  # (N,)
    total_len = cum_lens[-1]

    for _ in range(n_steps):
        # --- Pure pursuit look-ahead distance (from BridgeDrive agent) ---
        aim_dist = np.clip(0.975 * v + 1.915, 2.4, 10.5)

        # --- Find aim point on route in world frame ---
        aim_point_ego = _interpolate_route(route, cum_lens, total_len, x, y, yaw, aim_dist)

        # --- Heading error to aim point (in current vehicle frame) ---
        dx = aim_point_ego[0] - x
        dy = aim_point_ego[1] - y
        # Rotate into vehicle frame
        cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
        local_x =  cos_y * dx - sin_y * dy
        local_y =  sin_y * dx + cos_y * dy

        alpha = np.arctan2(local_y, local_x)

        # --- Pure pursuit steering angle ---
        d = max(np.sqrt(dx**2 + dy**2), 1e-3)
        steer_angle = np.arctan(2.0 * wheelbase * np.sin(alpha) / d)
        steer_angle = np.clip(steer_angle, -max_steer_angle, max_steer_angle)

        # --- Bicycle model integration ---
        x   += v * np.cos(yaw) * dt
        y   += v * np.sin(yaw) * dt
        yaw += v * np.tan(steer_angle) / wheelbase * dt

        traj.append([x, y])

    return np.array(traj, dtype=np.float32)   # (n_steps, 2)


def _interpolate_route(
    route: np.ndarray,
    cum_lens: np.ndarray,
    total_len: float,
    cur_x: float,
    cur_y: float,
    cur_yaw: float,
    aim_dist: float,
) -> np.ndarray:
    """
    Find a point on `route` that is `aim_dist` metres ahead of the current
    vehicle position (cur_x, cur_y) along the route arc.

    Strategy: find the closest route point to current position, then walk
    `aim_dist` further along the arc.  Clamps to the route end if needed.
    """
    # Closest route point index
    dists_to_cur = np.linalg.norm(route - np.array([cur_x, cur_y]), axis=1)
    closest_idx = int(np.argmin(dists_to_cur))

    # Target arc length
    target_arc = cum_lens[closest_idx] + aim_dist
    target_arc = min(target_arc, total_len)

    # Interpolate
    idx = np.searchsorted(cum_lens, target_arc, side='right') - 1
    idx = int(np.clip(idx, 0, len(route) - 2))
    seg_len = cum_lens[idx + 1] - cum_lens[idx]
    if seg_len < 1e-6:
        return route[idx]
    t = (target_arc - cum_lens[idx]) / seg_len
    return route[idx] * (1 - t) + route[idx + 1] * t
