#!/usr/bin/env python3
"""Visualize NAVSIM navtest drivable_area_compliance==0 cases."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from shapely import affinity
from shapely.geometry import LineString, Polygon
from tqdm import tqdm

from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array
from navsim_motdp.agents.cached_diffusion_agent import CachedDiffusionAgent


CSV_PATH = "/workspace2/z_project/navsim_exp_motdp/motdp_cached_navtest_npy_full_e60/2026.05.16.09.25.04/2026.05.16.10.31.04.csv"
CACHE_DIR = "/workspace2/z_project/motdp_bev_cache_navtest_npy"
CHECKPOINT_PATH = "/workspace2/z_project/motdp_logs/navsim_simple_official_lr1e4_e60_b64/best_model.pt"
METRIC_CACHE_PATH = "/workspace2/data/navsim/processed_data/metric_cache_navtest"
OUTPUT_DIR = "/workspace2/z_project/navsim_debug/dac0_bev_full_e60_20260516"


LAYER_STYLE = {
    "ROADBLOCK": ("#d9dde5", 0.22, "#c5cbd6", 0.35),
    "ROADBLOCK_CONNECTOR": ("#d9dde5", 0.18, "#c5cbd6", 0.3),
    "LANE": ("#eef2f6", 0.62, "#9aa4b2", 0.65),
    "LANE_CONNECTOR": ("#fff0a8", 0.72, "#d99b00", 0.88),
    "INTERSECTION": ("#dcd2ff", 0.42, "#8b73d6", 0.65),
    "DRIVABLE_AREA": ("#d7f0d2", 0.28, "#8ac184", 0.45),
    "CARPARK_AREA": ("#eee4ce", 0.32, "#b09d79", 0.45),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--metric-cache", default=METRIC_CACHE_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--max-cases", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--random-sample", action="store_true")
    parser.add_argument("--contact-sheet", type=int, default=64)
    parser.add_argument("--margin-x", type=float, default=70.0)
    parser.add_argument("--margin-y", type=float, default=44.0)
    return parser.parse_args()


def transform_geometry_to_local(geometry, origin):
    translated = affinity.affine_transform(geometry, [1, 0, 0, 1, -origin.x, -origin.y])
    c, s = math.cos(origin.heading), math.sin(origin.heading)
    return affinity.affine_transform(translated, [c, s, -s, c, 0, 0])


def points_global_to_local(points, origin):
    points = np.asarray(points, dtype=np.float64)
    dx = points[..., 0] - origin.x
    dy = points[..., 1] - origin.y
    c, s = math.cos(origin.heading), math.sin(origin.heading)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    return np.stack([local_x, local_y], axis=-1)


def plot_xy(ax, xy, *args, label=None, **kwargs):
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) == 0:
        return
    ax.plot(xy[:, 1], xy[:, 0], *args, label=label, **kwargs)


def scatter_xy(ax, xy, *args, label=None, **kwargs):
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) == 0:
        return
    ax.scatter(xy[:, 1], xy[:, 0], *args, label=label, **kwargs)


def plot_polygon(ax, geometry, fill_color, fill_alpha, edge_color, edge_alpha, linewidth=0.7, zorder=1):
    if geometry.is_empty:
        return
    if geometry.geom_type == "MultiPolygon":
        for geom in geometry.geoms:
            plot_polygon(ax, geom, fill_color, fill_alpha, edge_color, edge_alpha, linewidth, zorder)
        return
    if not isinstance(geometry, Polygon):
        return
    x, y = geometry.exterior.xy
    ax.fill(y, x, color=fill_color, alpha=fill_alpha, linewidth=0, zorder=zorder)
    ax.plot(y, x, color=edge_color, alpha=edge_alpha, linewidth=linewidth, zorder=zorder + 0.1)


def plot_linestring(ax, geometry, *args, **kwargs):
    if geometry.is_empty:
        return
    if geometry.geom_type == "MultiLineString":
        for geom in geometry.geoms:
            plot_linestring(ax, geom, *args, **kwargs)
        return
    if not isinstance(geometry, LineString):
        return
    x, y = geometry.xy
    ax.plot(y, x, *args, **kwargs)


def trajectory_to_simulated_states(metric_cache, trajectory):
    sampling = trajectory.trajectory_sampling
    pred_interp = transform_trajectory(trajectory, metric_cache.ego_state)
    pred_states = get_trajectory_as_array(pred_interp, sampling, metric_cache.ego_state.time_point)
    simulator = PDMSimulator(proposal_sampling=sampling)
    return simulator.simulate_proposals(pred_states[None, ...], metric_cache.ego_state)[0]


def compute_offroad_mask(metric_cache, simulated_states):
    coords = state_array_to_coords_array(
        simulated_states[None, ...],
        metric_cache.ego_state.car_footprint.vehicle_parameters,
    )[0]
    in_polygons = metric_cache.drivable_area_map.points_in_polygons(coords)
    drivable_area_indices = metric_cache.drivable_area_map.get_indices_of_map_type(
        [
            SemanticMapLayer.ROADBLOCK,
            SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.DRIVABLE_AREA,
            SemanticMapLayer.CARPARK_AREA,
        ]
    )
    corners_in_any_drivable = in_polygons[drivable_area_indices, :, :4].sum(axis=0) > 0
    offroad_mask = corners_in_any_drivable.sum(axis=-1) < 4
    return coords, offroad_mask


def draw_case(ax, metric_cache, pred_traj, row, args):
    origin = metric_cache.ego_state.rear_axle

    # Map polygons and route highlight.
    route_tokens = set(metric_cache.route_lane_ids or [])
    for token, map_type, geom in zip(
        metric_cache.drivable_area_map.tokens,
        metric_cache.drivable_area_map.map_types,
        metric_cache.drivable_area_map._geometries,
    ):
        local_geom = transform_geometry_to_local(geom, origin)
        name = getattr(map_type, "name", str(map_type))
        fill, fill_alpha, edge, edge_alpha = LAYER_STYLE.get(name, ("#eeeeee", 0.25, "#999999", 0.4))
        plot_polygon(ax, local_geom, fill, fill_alpha, edge, edge_alpha, linewidth=0.55, zorder=1)
        if token in route_tokens:
            plot_polygon(ax, local_geom, "none", 0.0, "#159947", 0.95, linewidth=1.5, zorder=2)

    # Centerline / PDM path.
    try:
        centerline_local = transform_geometry_to_local(metric_cache.centerline.linestring, origin)
        plot_linestring(ax, centerline_local, color="#111111", linestyle="--", linewidth=1.7, alpha=0.9, zorder=5, label="centerline")
    except Exception:
        pass

    # Human trajectory if available in metric cache.
    if metric_cache.human_trajectory is not None:
        human_xy = np.asarray(metric_cache.human_trajectory.poses[:, :2], dtype=np.float64)
        human_xy = np.concatenate([np.zeros((1, 2)), human_xy], axis=0)
        plot_xy(ax, human_xy, color="#2ca02c", linewidth=2.2, marker="o", markersize=3.0, zorder=8, label="human")

    # Raw model trajectory in ego frame and y-mirrored hypothesis.
    pred_xy = np.asarray(pred_traj.poses[:, :2], dtype=np.float64)
    pred_with_origin = np.concatenate([np.zeros((1, 2)), pred_xy], axis=0)
    plot_xy(ax, pred_with_origin, color="#d62728", linewidth=2.4, marker="o", markersize=3.2, zorder=9, label="model raw")
    mirror_xy = pred_with_origin.copy()
    mirror_xy[:, 1] *= -1.0
    plot_xy(ax, mirror_xy, color="#bf00ff", linestyle=":", linewidth=1.8, marker="x", markersize=3.0, zorder=7, label="model y-flip")

    # Simulated states and off-road footprint corners used by DAC.
    simulated_states = trajectory_to_simulated_states(metric_cache, pred_traj)
    ego_coords, offroad_mask = compute_offroad_mask(metric_cache, simulated_states)
    sim_local = points_global_to_local(simulated_states[:, :2], origin)
    plot_xy(ax, sim_local, color="#1f77b4", linewidth=1.5, alpha=0.85, zorder=8, label="simulated")
    scatter_xy(ax, sim_local[~offroad_mask], s=16, color="#1f77b4", zorder=10)
    scatter_xy(ax, sim_local[offroad_mask], s=34, color="#ff0000", edgecolors="#ffffff", linewidths=0.5, zorder=11, label="offroad step")

    # Footprint outlines for each simulated step.
    for step_idx, corners_global in enumerate(ego_coords[:, :4, :]):
        corners_local = points_global_to_local(np.asarray(corners_global), origin)
        closed = np.concatenate([corners_local, corners_local[:1]], axis=0)
        color = "#ff0000" if offroad_mask[step_idx] else "#1f77b4"
        plot_xy(ax, closed, color=color, alpha=0.35 if offroad_mask[step_idx] else 0.18, linewidth=1.0, zorder=6)

    ax.scatter([0], [0], marker="*", s=140, color="#000000", zorder=12, label="ego")
    ax.set_title(
        f"DAC=0 token={row.token} score={row.score:.3f} "
        f"ego_prog={row.ego_progress:.2f} lane={row.lane_keeping:.0f} off_steps={int(offroad_mask.sum())}",
        fontsize=9,
    )
    ax.set_xlabel("local y / lateral (left +)")
    ax.set_ylabel("local x / forward")
    ax.set_aspect("equal")
    ax.set_xlim(args.margin_y / 2, -args.margin_y / 2)
    ax.set_ylim(-args.margin_x * 0.20, args.margin_x * 0.80)
    ax.grid(True, color="#cccccc", linewidth=0.35, alpha=0.5)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.86)
    return {
        "offroad_steps": int(offroad_mask.sum()),
        "first_offroad_step": int(np.where(offroad_mask)[0][0]) if offroad_mask.any() else -1,
        "pred_final_x": float(pred_xy[-1, 0]),
        "pred_final_y": float(pred_xy[-1, 1]),
        "sim_final_x": float(sim_local[-1, 0]),
        "sim_final_y": float(sim_local[-1, 1]),
    }


def make_contact_sheet(image_paths, output_path, thumb_size=(360, 360), cols=8):
    image_paths = list(image_paths)
    if not image_paths:
        return
    rows = math.ceil(len(image_paths) / cols)
    sheet = Image.new("RGB", (cols * thumb_size[0], rows * thumb_size[1]), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, path in enumerate(image_paths):
        img = Image.open(path).convert("RGB")
        img.thumbnail((thumb_size[0], thumb_size[1] - 24))
        x = (idx % cols) * thumb_size[0]
        y = (idx // cols) * thumb_size[1]
        sheet.paste(img, (x + (thumb_size[0] - img.width) // 2, y + 18))
        draw.text((x + 4, y + 2), Path(path).stem[:42], fill=(0, 0, 0))
    sheet.save(output_path)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    img_dir = out_dir / "png"
    img_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df[df["token"] != "average_all_frames"].copy()
    bad = df[df["drivable_area_compliance"] == 0].copy().reset_index(drop=True)
    if args.random_sample:
        bad = bad.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.max_cases > 0:
        bad = bad.head(args.max_cases)

    print(f"DAC=0 cases selected: {len(bad)}", flush=True)
    print(f"Output: {out_dir}", flush=True)

    agent = CachedDiffusionAgent(
        checkpoint_path=args.checkpoint,
        cache_dir=args.cache_dir,
        device=args.device,
        fallback="raise",
    )
    agent.initialize()
    metric_loader = MetricCacheLoader(Path(args.metric_cache))

    records = []
    rendered_paths = []
    dummy_input = SimpleNamespace(ego_statuses=[])
    for row in tqdm(list(bad.itertuples(index=False)), desc="Rendering DAC=0 BEV"):
        token = str(row.token)
        metric_cache = metric_loader.get_from_token(token)
        scene = SimpleNamespace(scene_metadata=SimpleNamespace(initial_token=token, num_history_frames=4), frames=[])
        pred_traj = agent.compute_trajectory(dummy_input, scene)

        fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=140)
        extra = draw_case(ax, metric_cache, pred_traj, row, args)
        png_path = img_dir / f"{token}_dac0.png"
        fig.tight_layout()
        fig.savefig(png_path)
        plt.close(fig)
        rendered_paths.append(png_path)

        record = row._asdict()
        record.update(extra)
        record["png"] = str(png_path)
        records.append(record)

    summary_csv = out_dir / "dac0_visualization_summary.csv"
    pd.DataFrame(records).to_csv(summary_csv, index=False)
    manifest = {
        "csv": args.csv,
        "cache_dir": args.cache_dir,
        "checkpoint": args.checkpoint,
        "metric_cache": args.metric_cache,
        "num_cases": len(records),
        "png_dir": str(img_dir),
        "summary_csv": str(summary_csv),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.contact_sheet > 0:
        make_contact_sheet(rendered_paths[: args.contact_sheet], out_dir / "contact_sheet_first_cases.png")

    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
