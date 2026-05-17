"""Wrapper: precompute official4cam train BEV cache."""
import sys, yaml
sys.path.insert(0, '/home/z/code/motdp_z_navsim_motdp/scripts')
import precompute_bev_cache as pc

pc.TRAINVAL_LOGS = '/workspace2/data/navsim/navsim_logs/trainval'
pc.SENSOR_BLOBS = '/workspace2/data/navsim/sensor_blobs/trainval'
pc.CACHE_DIR = '/workspace2/z_project/motdp_bev_cache_train_official4cam'

with open('/home/z/code/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml') as f:
    _nav = yaml.safe_load(f)
pc.NAVTRAIN_LOGS = set(_nav['log_names'])
pc.NAVTRAIN_TOKENS = set(_nav.get('tokens') or [])

if __name__ == '__main__':
    pc.main()
