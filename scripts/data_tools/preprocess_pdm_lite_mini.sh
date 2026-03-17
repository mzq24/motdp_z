python dataset/preprocess_pdm_lite_withoutvqa.py \
  --data-root /media/z/data/dataset/pdm_lite_mini \
  --out-dir /media/z/data/dataset/pdm_lite_mini \
  --tmp-dir data \
  --obs-horizon 4 \
  --action-horizon 6 \
  --sample-interval 1 \
  --hz-interval 2 \
  --workers 4 \
  --save-mode frame
