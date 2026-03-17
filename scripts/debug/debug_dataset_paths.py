"""
Quick debug script to inspect dataset paths on HPC.
Usage: python scripts/debug_dataset_paths.py

Checks:
  1. route_source.pt   (Phase 1 output from pack_source)
  2. route_features.pt (Phase 2 output from extract)
  3. transfuser_bev_scene.pkl
  4. Diagnoses WHY route_source.pt is missing (no lidar/, no rgb/, etc.)
"""
import pickle
import os
import gzip

SCRATCH = "/home/users/ntu/wh.huang/scratch"
DATASET_ROOT = f"{SCRATCH}/z_projects/dataset/pdm_lite"
TRAIN_PACKED = f"{SCRATCH}/z_projects/dataset/pdm_lite/tmp_data/train/samples_packed.pkl"
VAL_PACKED = f"{SCRATCH}/z_projects/dataset/pdm_lite/tmp_data/val/samples_packed.pkl"

# Step 1: Build route_name -> event_name mapping from disk
print("Scanning disk for route_name -> event_name mapping...")
route_to_event = {}
for event_entry in os.scandir(DATASET_ROOT):
    if not event_entry.is_dir():
        continue
    if event_entry.name in ('tmp_data', 'tmp_data_tg_next'):
        continue
    for route_entry in os.scandir(event_entry.path):
        if route_entry.is_dir():
            route_to_event[route_entry.name] = event_entry.name
print(f"Found {len(route_to_event)} routes on disk across events.")

# Step 2: Check which routes have route_source.pt / route_features.pt / scene pkl
routes_with_source = 0
routes_without_source = 0
routes_with_features = 0
routes_without_features = 0
routes_with_scene_pkl = 0
routes_without_scene_pkl = 0
missing_source_examples = []
missing_feat_examples = []
missing_scene_examples = []
missing_feat_routes_full = []  # ALL routes missing features (for repack)

for route_name, event_name in route_to_event.items():
    route_dir = os.path.join(DATASET_ROOT, event_name, route_name)
    source_path = os.path.join(route_dir, 'route_source.pt')
    feat_path = os.path.join(route_dir, 'transfuser_feature', 'route_features.pt')
    scene_path = os.path.join(route_dir, 'transfuser_bev_scene.pkl')

    if os.path.exists(source_path):
        routes_with_source += 1
    else:
        routes_without_source += 1
        if len(missing_source_examples) < 10:
            missing_source_examples.append(f"{event_name}/{route_name}")

    if os.path.exists(feat_path):
        routes_with_features += 1
    else:
        routes_without_features += 1
        missing_feat_routes_full.append(f"{event_name}/{route_name}")
        if len(missing_feat_examples) < 10:
            missing_feat_examples.append(f"{event_name}/{route_name}")

    if os.path.exists(scene_path):
        routes_with_scene_pkl += 1
    else:
        routes_without_scene_pkl += 1
        if len(missing_scene_examples) < 5:
            missing_scene_examples.append(f"{event_name}/{route_name}")

print(f"\n{'='*60}")
print(f"  route_source.pt (Phase 1)")
print(f"{'='*60}")
print(f"Routes WITH    route_source.pt: {routes_with_source}")
print(f"Routes WITHOUT route_source.pt: {routes_without_source}")
if missing_source_examples:
    print(f"  Missing examples (up to 10):")
    for ex in missing_source_examples:
        print(f"    - {ex}")

print(f"\n{'='*60}")
print(f"  route_features.pt (Phase 2)")
print(f"{'='*60}")
print(f"Routes WITH    route_features.pt: {routes_with_features}")
print(f"Routes WITHOUT route_features.pt: {routes_without_features}")
if missing_feat_examples:
    print(f"  Missing examples (up to 10):")
    for ex in missing_feat_examples:
        print(f"    - {ex}")

print(f"\n{'='*60}")
print(f"  transfuser_bev_scene.pkl")
print(f"{'='*60}")
print(f"Routes WITH    transfuser_bev_scene.pkl: {routes_with_scene_pkl}")
print(f"Routes WITHOUT transfuser_bev_scene.pkl: {routes_without_scene_pkl}")
if missing_scene_examples:
    print(f"  Missing examples: {missing_scene_examples}")

# Step 3: Diagnose WHY route_source.pt is missing
if routes_without_source > 0:
    print(f"\n{'='*60}")
    print(f"  Diagnosing {routes_without_source} routes missing route_source.pt")
    print(f"{'='*60}")
    diag = {'no_lidar_dir': [], 'no_rgb_dir': [], 'no_laz_files': [],
            'no_results_json': [], 'is_FAILED': [], 'unknown': []}
    for route_name, event_name in route_to_event.items():
        route_dir = os.path.join(DATASET_ROOT, event_name, route_name)
        source_path = os.path.join(route_dir, 'route_source.pt')
        if os.path.exists(source_path):
            continue
        tag = f"{event_name}/{route_name}"
        if route_name.startswith('FAILED_'):
            diag['is_FAILED'].append(tag)
        elif not os.path.isdir(os.path.join(route_dir, 'lidar')):
            diag['no_lidar_dir'].append(tag)
        elif not os.path.isdir(os.path.join(route_dir, 'rgb')):
            diag['no_rgb_dir'].append(tag)
        elif len(list(os.scandir(os.path.join(route_dir, 'lidar')))) == 0:
            diag['no_laz_files'].append(tag)
        elif not os.path.exists(os.path.join(route_dir, 'results.json.gz')):
            diag['no_results_json'].append(tag)
        else:
            # Route looks valid but route_source.pt is missing — pack_source probably didn't run
            n_laz = len([f for f in os.scandir(os.path.join(route_dir, 'lidar')) if f.name.endswith('.laz')])
            n_jpg = len([f for f in os.scandir(os.path.join(route_dir, 'rgb')) if f.name.endswith('.jpg')])
            diag['unknown'].append(f"{tag}  (lidar: {n_laz} .laz, rgb: {n_jpg} .jpg)")

    for reason, routes_list in diag.items():
        if routes_list:
            print(f"\n  [{reason}] ({len(routes_list)} routes):")
            for r in routes_list[:10]:
                print(f"    - {r}")
            if len(routes_list) > 10:
                print(f"    ... and {len(routes_list) - 10} more")

# Step 4: Write full list of missing-feature routes to file (for targeted repack)
# Filter out routes that are fundamentally broken (no lidar, no results.json)
repackable_routes = []
skipped_broken = []
for r in missing_feat_routes_full:
    event_name, route_name = r.split('/', 1)
    route_dir = os.path.join(DATASET_ROOT, event_name, route_name)
    has_lidar = os.path.isdir(os.path.join(route_dir, 'lidar'))
    has_rgb = os.path.isdir(os.path.join(route_dir, 'rgb'))
    has_results = os.path.exists(os.path.join(route_dir, 'results.json.gz'))
    if has_lidar and has_rgb and has_results:
        repackable_routes.append(r)
    else:
        skipped_broken.append(r)

if repackable_routes:
    out_path = os.path.join(os.path.dirname(__file__), 'missing_routes.txt')
    with open(out_path, 'w') as f:
        for r in sorted(repackable_routes):
            f.write(r + '\n')
    print(f"\nWrote {len(repackable_routes)} repackable routes to {out_path}")

# Final summary
print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  Total routes on disk:    {len(route_to_event)}")
print(f"  Routes with features:    {routes_with_features}")
print(f"  Missing features:        {routes_without_features}")
if skipped_broken:
    print(f"  ├─ Incomplete data (no results.json.gz / no lidar): {len(skipped_broken)}")
    print(f"  │   These routes had incomplete simulation runs and CANNOT be used for training.")
    print(f"  │   Safe to ignore.")
if repackable_routes:
    print(f"  └─ Repackable (data intact, just not packed):       {len(repackable_routes)}")
    print(f"      Run: qsub scripts/nscc_repack_missing.pbs")
else:
    print(f"  └─ Repackable: 0  (nothing to do!)")
    print(f"      All {len(skipped_broken)} missing routes are broken/incomplete — safe to ignore.")

# # Show a few examples
# for route_name, event_name in list(route_to_event.items())[:3]:
#     feat_dir = os.path.join(DATASET_ROOT, event_name, route_name, 'transfuser_feature')
#     feat_pt = os.path.join(feat_dir, 'route_features.pt')
#     src_pt = os.path.join(DATASET_ROOT, event_name, route_name, 'route_source.pt')
#     print(f"\n  {event_name}/{route_name}:")
#     print(f"    transfuser_feature/ exists={os.path.exists(feat_dir)}")
#     print(f"    route_features.pt   exists={os.path.exists(feat_pt)}")
#     print(f"    route_source.pt     exists={os.path.exists(src_pt)}")
#     if os.path.exists(feat_dir):
#         contents = os.listdir(feat_dir)[:5]
#         print(f"    transfuser_feature/ contents (first 5): {contents}")
#
# # Step 3: Check pkl samples
# for label, packed_path in [("TRAIN", TRAIN_PACKED), ("VAL", VAL_PACKED)]:
#     print(f"\n{'='*60}")
#     print(f"  {label}: {packed_path}")
#     print(f"{'='*60}")
#
#     if not os.path.exists(packed_path):
#         print(f"  [NOT FOUND]")
#         continue
#
#     with open(packed_path, 'rb') as f:
#         samples = pickle.load(f)
#     print(f"  Total samples: {len(samples)}")
#
#     # Show first 3 samples' fields
#     print(f"\n  --- First 3 samples ---")
#     for i, s in enumerate(samples[:3]):
#         print(f"  [{i}] event_name={s.get('event_name', '<MISSING>')}")
#         print(f"       route_name={s.get('route_name', '<MISSING>')}")
#         print(f"       frame_id={s.get('frame_id', '<MISSING>')}")
#
#         route_name = s.get('route_name', '')
#         if route_name and route_name in route_to_event:
#             event = route_to_event[route_name]
#             feat_pt = os.path.join(DATASET_ROOT, event, route_name, 'transfuser_feature', 'route_features.pt')
#             print(f"       -> disk event={event}")
#             print(f"       -> route_features.pt exists={os.path.exists(feat_pt)}")
#         print()
#
#     # Count how many samples can be matched to a route with features
#     matchable = 0
#     unmatchable_routes = set()
#     for s in samples:
#         rn = s.get('route_name', '')
#         if rn in route_to_event:
#             feat_pt = os.path.join(DATASET_ROOT, route_to_event[rn], rn, 'transfuser_feature', 'route_features.pt')
#             if os.path.exists(feat_pt):
#                 matchable += 1
#                 continue
#         unmatchable_routes.add(rn)
#
#     print(f"  Samples matchable to route_features.pt: {matchable}/{len(samples)}")
#     print(f"  Unmatchable unique routes: {len(unmatchable_routes)}")
#     if unmatchable_routes:
#         for r in list(unmatchable_routes)[:5]:
#             in_disk = r in route_to_event
#             print(f"    {r}  (on disk={in_disk})")
#
#     del samples
