python model/transfuser_extractor/preprocess_dataset.py \
  --dataset_path /media/z/data/dataset/pdm_lite_mini \
  --config_path /media/z/data/models/garage2/pretrained_models/all_towns \
  --model_path /media/z/data/models/garage2/pretrained_models/all_towns/model_0030_1.pth \
  --no_skip_existing \
  --batch_size 16 \
  --device cuda:0 \
  --mode pack_and_extract
