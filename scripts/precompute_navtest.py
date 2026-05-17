"""Wrapper: precompute BEV cache for navtest split."""
import sys, yaml
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (str(_REPO_ROOT), str(_SCRIPT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
# Override module-level configs before importing
import precompute_bev_cache as pc

# Override paths for navtest
pc.TRAINVAL_LOGS = '/workspace2/data/navsim/navsim_logs/test'
pc.SENSOR_BLOBS = '/workspace2/data/navsim/sensor_blobs/test'
pc.CACHE_DIR = '/workspace2/z_project/motdp_bev_cache_navtest'

# Load navtest filter
with open('/home/z/code/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml') as f:
    _nav = yaml.safe_load(f)
pc.NAVTRAIN_LOGS = set(_nav['log_names'])
pc.NAVTRAIN_TOKENS = set(_nav.get('tokens') or [])

if __name__ == '__main__':
    pc.main()
