python dataset/preprocess_pdm_lite.py \
  --data-root /media/z/data/dataset/carla/PDM_Lite/ \
  --out-dir /media/z/data/dataset/carla/PDM_Lite_processed/ \
  --tmp-dir data \
  --obs-horizon 2 \
  --action-horizon 8 \
  --sample-interval 2 \
  --workers 4 \
  --save-mode frame