"""
Check total size of all route_features.pt files and estimate memory usage.
Usage: python scripts/check_feature_size.py
"""
import os

SCRATCH = "/home/users/ntu/wh.huang/scratch"
DATASET_ROOT = f"{SCRATCH}/z_projects/dataset/pdm_lite"

total_bytes = 0
count = 0
total_frames = 0

for event_entry in os.scandir(DATASET_ROOT):
    if not event_entry.is_dir() or event_entry.name in ('tmp_data', 'tmp_data_tg_next'):
        continue
    for route_entry in os.scandir(event_entry.path):
        if not route_entry.is_dir():
            continue
        feat_path = os.path.join(route_entry.path, 'transfuser_feature', 'route_features.pt')
        if os.path.exists(feat_path):
            sz = os.path.getsize(feat_path)
            total_bytes += sz
            count += 1

print(f"route_features.pt files found: {count}")
print(f"Total size on disk:  {total_bytes / 1e9:.2f} GB")
print(f"Average per route:   {total_bytes / max(count, 1) / 1e6:.1f} MB")
print()
print(f"Estimated in-memory (float32): {total_bytes / 1e9:.2f} GB")
print(f"Estimated in-memory (float16): {total_bytes / 2 / 1e9:.2f} GB")
print(f"  (torch.save adds overhead, actual tensor data is slightly less than file size)")
