"""
修复脚本：删除 bad_routes.txt 中列出的 route 的 route_source.pt 和 route_features.pt，
以便重新跑 pack_source + extract PBS 任务时自动重新生成。

用法:
  # Dry run (只打印，不删除)
  python scripts/fix_bad_routes.py

  # 实际删除
  python scripts/fix_bad_routes.py --delete

  # 使用自定义 bad_routes.txt
  python scripts/fix_bad_routes.py --bad_routes /path/to/bad_routes.txt --delete
"""

import os
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Delete bad route_source.pt and route_features.pt for re-extraction')
    parser.add_argument('--bad_routes', type=str,
                        default='/home/users/ntu/wh.huang/scratch/z_projects/dataset/pdm_lite/bad_routes.txt',
                        help='Path to bad_routes.txt (one route dir per line)')
    parser.add_argument('--delete', action='store_true',
                        help='Actually delete files. Without this flag, only prints what would be deleted.')
    args = parser.parse_args()

    bad_routes_file = Path(args.bad_routes)
    if not bad_routes_file.exists():
        print(f"ERROR: bad_routes.txt not found: {bad_routes_file}")
        sys.exit(1)

    with open(bad_routes_file, 'r') as f:
        routes = [line.strip() for line in f if line.strip()]

    print(f"Found {len(routes)} bad routes in {bad_routes_file}")
    if not args.delete:
        print("*** DRY RUN — add --delete to actually remove files ***\n")

    deleted_source = 0
    deleted_features = 0
    missing_source = 0
    missing_features = 0

    for route_path in routes:
        route_dir = Path(route_path)
        source_pt = route_dir / 'route_source.pt'
        features_pt = route_dir / 'transfuser_feature' / 'route_features.pt'

        # route_source.pt
        if source_pt.exists():
            size_mb = source_pt.stat().st_size / 1e6
            if args.delete:
                source_pt.unlink()
                print(f"  DELETED route_source.pt ({size_mb:.1f} MB): {source_pt}")
            else:
                print(f"  WOULD DELETE route_source.pt ({size_mb:.1f} MB): {source_pt}")
            deleted_source += 1
        else:
            missing_source += 1

        # route_features.pt
        if features_pt.exists():
            size_mb = features_pt.stat().st_size / 1e6
            if args.delete:
                features_pt.unlink()
                print(f"  DELETED route_features.pt ({size_mb:.1f} MB): {features_pt}")
            else:
                print(f"  WOULD DELETE route_features.pt ({size_mb:.1f} MB): {features_pt}")
            deleted_features += 1
        else:
            missing_features += 1

    print(f"\n{'=' * 50}")
    action = "Deleted" if args.delete else "Would delete"
    print(f"{action} route_source.pt:   {deleted_source} (missing: {missing_source})")
    print(f"{action} route_features.pt: {deleted_features} (missing: {missing_features})")

    if args.delete:
        print(f"\nNow re-run PBS jobs to regenerate:")
        print(f"  1. qsub scripts/nscc_pack_source.pbs    # Phase 1: pack source")
        print(f"  2. qsub scripts/nscc_extract_features.pbs  # Phase 2: extract features (after Phase 1 done)")
        print(f"  3. python scripts/check_route_features_framenums.py  # Verify")
    else:
        print(f"\nRe-run with --delete to actually remove files.")


if __name__ == '__main__':
    main()
