#!/bin/bash

# TransFuser 推理时间测试脚本

echo "Testing TransFuser Inference Speed with Batch Size 64"
echo "======================================================"

python model/transfuser_extractor/benchmark_inference.py \
  --config_path /media/z/data/models/garage2/pretrained_models/all_towns \
  --batch_size 64 \
  --num_iterations 100 \
  --warmup_iterations 10 \
  --device cuda:0

echo ""
echo "======================================================"
echo "Test completed!"
