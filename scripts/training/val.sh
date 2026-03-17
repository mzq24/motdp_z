# -m debugpy --listen 0.0.0.0:5678 --wait-for-client
python training/train_carla_bev.py \
    --config_path config/pdm_local.yaml \
    --resume ./checkpoints/base_plus_mot_anchor_norm/dit_policy_best.pt \
    --val_only