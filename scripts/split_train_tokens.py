import yaml, sys, os
sys.path.insert(0, '/home/z/code/navsim')
from pathlib import Path
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader

with open('/home/z/code/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml') as f:
    cfg = yaml.safe_load(f)

sf = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True,
                 log_names=cfg['log_names'], tokens=cfg.get('tokens'))
sf.frame_interval = 1

print('Scanning pkls...')
loader = SceneLoader(
    data_path=Path('/workspace2/data/navsim/navsim_logs/trainval'),
    original_sensor_path=Path('/workspace2/data/navsim/sensor_blobs/trainval_full'),
    scene_filter=sf, sensor_config=None,
)

tokens = loader.tokens
n_total = len(tokens)
n_per = (n_total + 5) // 6
print('Total tokens: {}, per GPU: ~{}'.format(n_total, n_per))

out_dir = '/workspace2/z_project/motdp_bev_cache_train_official4cam'
os.makedirs(out_dir, exist_ok=True)

for gpu in range(6):
    start = gpu * n_per
    end = min(start + n_per, n_total)
    chunk = tokens[start:end]
    path = '{}/tokens_gpu{}.txt'.format(out_dir, gpu)
    with open(path, 'w') as f:
        for t in chunk:
            f.write(t + '\n')
    print('GPU {}: {} tokens -> {}'.format(gpu, len(chunk), path))
print('Done')
