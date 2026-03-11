"""
诊断脚本：检查所有 route_features.pt 的 frame_nums 是否与 lidar 文件名一致。

正常情况下 frame_nums 应该与 sorted(lidar/*.laz) 的 stem 完全一致。
本脚本找出不一致的 route，输出：
  1. 总体统计
  2. 每个异常 route 的详情
  3. 需要重新提取的 route 列表（写入 bad_routes.txt）

用法:
  python scripts/check_route_features_framenums.py --dataset_root /path/to/pdm_lite
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from tqdm import tqdm


def check_route(route_dir: Path):
    """检查单个 route 的 frame_nums 是否正确。
    
    Returns:
        None if no route_features.pt exists
        dict with diagnosis info otherwise
    """
    feat_path = route_dir / 'transfuser_feature' / 'route_features.pt'
    if not feat_path.exists():
        return None

    lidar_dir = route_dir / 'lidar'
    
    # Load frame_nums from route_features.pt
    try:
        pack = torch.load(feat_path, weights_only=True, map_location='cpu')
        feat_frame_nums = pack['frame_nums']
    except Exception as e:
        return {
            'status': 'load_error',
            'error': str(e),
            'route': str(route_dir),
        }

    # Get expected frame_nums from lidar files
    if lidar_dir.exists():
        lidar_nums = sorted([f.stem for f in lidar_dir.glob('*.laz')])
    else:
        lidar_nums = None

    # Check measurement files too
    meas_dir = route_dir / 'measurements'
    if meas_dir.exists():
        meas_nums = sorted([f.name.split('.')[0] for f in meas_dir.glob('*.json.gz')])
    else:
        meas_nums = None

    result = {
        'status': 'ok',
        'route': str(route_dir),
        'n_feat_frames': len(feat_frame_nums),
        'feat_first': feat_frame_nums[0] if feat_frame_nums else '(empty)',
        'feat_last': feat_frame_nums[-1] if feat_frame_nums else '(empty)',
    }

    if lidar_nums is not None:
        result['n_lidar'] = len(lidar_nums)
        result['lidar_first'] = lidar_nums[0] if lidar_nums else '(empty)'
        result['lidar_last'] = lidar_nums[-1] if lidar_nums else '(empty)'

        if feat_frame_nums == lidar_nums:
            result['status'] = 'ok'
        elif len(feat_frame_nums) == len(lidar_nums) and feat_frame_nums != lidar_nums:
            # Same count but different names — likely a version mismatch
            result['status'] = 'name_mismatch'
            # Check if it's a simple offset
            try:
                feat_ints = [int(x) for x in feat_frame_nums]
                lidar_ints = [int(x) for x in lidar_nums]
                offsets = [f - l for f, l in zip(feat_ints, lidar_ints)]
                if len(set(offsets)) == 1:
                    result['offset'] = offsets[0]
                else:
                    result['offset'] = 'variable'
            except ValueError:
                result['offset'] = 'non-numeric'
        elif len(feat_frame_nums) != len(lidar_nums):
            result['status'] = 'count_mismatch'
    else:
        result['n_lidar'] = 0
        result['status'] = 'no_lidar_dir'

    return result


def main():
    parser = argparse.ArgumentParser(description='Check route_features.pt frame_nums consistency')
    parser.add_argument('--dataset_root', type=str,
                        default='/home/users/ntu/wh.huang/scratch/z_projects/dataset/pdm_lite')
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"ERROR: dataset_root does not exist: {dataset_root}")
        sys.exit(1)

    # Scan all routes
    print(f"Scanning {dataset_root} ...")
    routes = []
    for event_entry in sorted(dataset_root.iterdir()):
        if not event_entry.is_dir() or event_entry.name in ('tmp_data', 'tmp_data_tg_next'):
            continue
        for route_entry in sorted(event_entry.iterdir()):
            if not route_entry.is_dir():
                continue
            routes.append(route_entry)

    print(f"Found {len(routes)} route directories.\n")

    # Check each route
    total = 0
    ok = 0
    bad = []
    no_features = 0
    errors = []

    for route_dir in tqdm(routes, desc="Checking routes"):
        result = check_route(route_dir)
        if result is None:
            no_features += 1
            continue

        total += 1
        if result['status'] == 'ok':
            ok += 1
        elif result['status'] == 'load_error':
            errors.append(result)
        else:
            bad.append(result)

    # Print summary
    print("=" * 70)
    print(f"SUMMARY")
    print(f"=" * 70)
    print(f"Total routes with route_features.pt:  {total}")
    print(f"  OK (frame_nums match lidar):        {ok}")
    print(f"  BAD (mismatch):                     {len(bad)}")
    print(f"  Load errors:                        {len(errors)}")
    print(f"Routes without route_features.pt:     {no_features}")
    print()

    if bad:
        print("=" * 70)
        print(f"BAD ROUTES ({len(bad)}):")
        print("=" * 70)
        for r in bad:
            rel = os.path.relpath(r['route'], dataset_root)
            print(f"\n  Route: {rel}")
            print(f"    Status:      {r['status']}")
            print(f"    feat frames: {r['n_feat_frames']}  [{r['feat_first']} .. {r['feat_last']}]")
            if 'n_lidar' in r:
                print(f"    lidar files: {r['n_lidar']}  [{r.get('lidar_first', '?')} .. {r.get('lidar_last', '?')}]")
            if 'offset' in r:
                print(f"    offset:      {r['offset']}")

    if errors:
        print(f"\nLOAD ERRORS ({len(errors)}):")
        for r in errors:
            print(f"  {r['route']}: {r['error']}")

    # Write bad routes list
    output_file = dataset_root / 'bad_routes.txt'
    with open(output_file, 'w') as f:
        for r in bad:
            f.write(r['route'] + '\n')
    print(f"\nBad routes list written to: {output_file}")

    # Also write a summary of unique offsets
    if bad:
        offsets = set()
        for r in bad:
            if 'offset' in r:
                offsets.add(str(r['offset']))
        if offsets:
            print(f"\nUnique offsets found: {offsets}")


if __name__ == '__main__':
    main()
