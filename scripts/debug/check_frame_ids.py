"""
Check what frame_ids/frame_nums actually contain in both file formats.
Usage: python scripts/check_frame_ids.py --dataset_path /path/to/pdm_lite
"""
import argparse
import os
import pickle
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True)
    args = parser.parse_args()

    # Find a route with both files
    for event_entry in os.scandir(args.dataset_path):
        if not event_entry.is_dir() or event_entry.name in ('tmp_data', 'tmp_data_tg_next'):
            continue
        for route_entry in os.scandir(event_entry.path):
            if not route_entry.is_dir():
                continue
            rf_path = os.path.join(route_entry.path, 'transfuser_feature', 'route_features.pt')
            sp_path = os.path.join(route_entry.path, 'transfuser_bev_scene.pkl')
            if os.path.exists(rf_path) and os.path.exists(sp_path):
                print(f"Route: {event_entry.name}/{route_entry.name}")

                rf = torch.load(rf_path, weights_only=True)
                with open(sp_path, 'rb') as f:
                    sp = pickle.load(f)

                rf_fns = rf['frame_nums']
                sp_fids = sp['frame_ids']

                print(f"\n  route_features.pt frame_nums:")
                print(f"    type: {type(rf_fns)}, len: {len(rf_fns)}")
                print(f"    first 10: {rf_fns[:10]}")
                print(f"    last 5:   {rf_fns[-5:]}")
                print(f"    elem type: {type(rf_fns[0])}")

                print(f"\n  scene_pkl frame_ids:")
                print(f"    type: {type(sp_fids)}, len: {len(sp_fids)}")
                print(f"    first 10: {sp_fids[:10]}")
                print(f"    last 5:   {sp_fids[-5:]}")
                print(f"    elem type: {type(sp_fids[0])}")

                # Check if they represent the same frames
                rf_set = set(rf_fns)
                sp_set = set(f'{int(fid):04d}' for fid in sp_fids)
                sp_set_raw = set(str(fid) for fid in sp_fids)
                print(f"\n  Overlap (zero-padded): {len(rf_set & sp_set)} / rf={len(rf_set)}, sp={len(sp_set)}")
                print(f"  Overlap (raw str):     {len(rf_set & sp_set_raw)} / rf={len(rf_set)}, sp={len(sp_set_raw)}")

                # Are they identical sets?
                if rf_set == sp_set:
                    print("  -> IDENTICAL frame sets (zero-padded)")
                elif rf_set == sp_set_raw:
                    print("  -> IDENTICAL frame sets (raw str)")
                else:
                    only_rf = sorted(rf_set - sp_set)[:5]
                    only_sp = sorted(sp_set - rf_set)[:5]
                    print(f"  -> DIFFERENT! only in rf: {only_rf}, only in sp: {only_sp}")

                return


if __name__ == '__main__':
    main()
