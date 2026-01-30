"""
TransFuser Backbone 推理时间测试
=================================
测试不同 batch size 下的推理速度

使用方法:
    python benchmark_inference.py --config_path /path/to/config \
                                  --batch_size 64 \
                                  --num_iterations 100 \
                                  --device cuda:0
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import numpy as np

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from backbone_extractor import TransFuserBackboneExtractor


class InferenceBenchmark:
    """
    推理性能测试器
    """
    
    def __init__(self, config_path: str, batch_size: int = 1, device: str = 'cuda:0'):
        """
        初始化测试器
        
        Args:
            config_path: TransFuser 配置目录
            batch_size: 批处理大小
            device: 运行设备
        """
        self.config_path = config_path
        self.batch_size = batch_size
        self.device = device
        
        print("=" * 70)
        print("TransFuser Backbone Inference Benchmark")
        print("=" * 70)
        print(f"Config path: {config_path}")
        print(f"Batch size: {batch_size}")
        print(f"Device: {device}")
        print()
        
        # 创建提取器
        print("Initializing TransFuser Backbone Extractor...")
        self.extractor = TransFuserBackboneExtractor(
            config_path=config_path,
            device=device
        )
        print("Extractor initialized successfully!\n")
        
    def create_dummy_inputs(self):
        """
        创建 dummy 输入张量
        """
        config = self.extractor.config
        
        # RGB: [B, 3, H, W]
        rgb_height = config.cropped_height if config.crop_image else config.camera_height
        rgb_width = config.cropped_width if config.crop_image else config.camera_width
        dummy_rgb = torch.randn(
            self.batch_size, 3, rgb_height, rgb_width,
            dtype=torch.float32,
            device=self.device
        ) * 255
        dummy_rgb = dummy_rgb.clamp(0, 255)
        
        # LiDAR BEV: [B, C, H, W]
        lidar_channels = 2 if config.use_ground_plane else 1
        lidar_channels *= config.lidar_seq_len
        dummy_lidar = torch.randn(
            self.batch_size, lidar_channels,
            config.lidar_resolution_height,
            config.lidar_resolution_width,
            dtype=torch.float32,
            device=self.device
        )
        
        print(f"Input shapes:")
        print(f"  RGB: {dummy_rgb.shape}")
        print(f"  LiDAR BEV: {dummy_lidar.shape}\n")
        
        return dummy_rgb, dummy_lidar
    
    def benchmark(self, num_iterations: int = 100, warmup_iterations: int = 10):
        """
        运行推理时间测试
        
        Args:
            num_iterations: 测试迭代次数
            warmup_iterations: 预热迭代次数
        """
        print(f"Starting benchmark with {num_iterations} iterations...")
        print(f"Warmup iterations: {warmup_iterations}\n")
        
        # 创建 dummy 输入
        dummy_rgb, dummy_lidar = self.create_dummy_inputs()
        
        # 预热 GPU
        print("Warming up GPU...")
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = self.extractor(dummy_rgb, dummy_lidar)
        
        # 同步 GPU
        torch.cuda.synchronize(self.device)
        
        # 开始测试
        print(f"Running {num_iterations} iterations...\n")
        times = []
        
        with torch.no_grad():
            for i in range(num_iterations):
                # GPU 同步
                torch.cuda.synchronize(self.device)
                start_time = time.perf_counter()
                
                # 前向传播
                output = self.extractor(dummy_rgb, dummy_lidar)
                
                # GPU 同步
                torch.cuda.synchronize(self.device)
                end_time = time.perf_counter()
                
                elapsed_time = (end_time - start_time) * 1000  # 转换为 ms
                times.append(elapsed_time)
                
                if (i + 1) % max(1, num_iterations // 10) == 0:
                    print(f"  Iteration {i + 1}/{num_iterations}: {elapsed_time:.2f} ms")
        
        # 计算统计信息
        times_np = np.array(times)
        print("\n" + "=" * 70)
        print("Benchmark Results")
        print("=" * 70)
        print(f"Batch size: {self.batch_size}")
        print(f"Total iterations: {num_iterations}")
        print()
        print("Timing Statistics (ms):")
        print(f"  Mean:     {times_np.mean():.2f}")
        print(f"  Std:      {times_np.std():.2f}")
        print(f"  Min:      {times_np.min():.2f}")
        print(f"  Max:      {times_np.max():.2f}")
        print(f"  Median:   {np.median(times_np):.2f}")
        print(f"  P95:      {np.percentile(times_np, 95):.2f}")
        print(f"  P99:      {np.percentile(times_np, 99):.2f}")
        print()
        print("Throughput:")
        avg_time_per_batch = times_np.mean()
        throughput_fps = (self.batch_size / (avg_time_per_batch / 1000))
        throughput_samples_per_sec = throughput_fps
        print(f"  {throughput_fps:.2f} samples/sec")
        print(f"  {1000 / avg_time_per_batch:.2f} batches/sec")
        print("=" * 70)
        
        return times_np
    
    def benchmark_different_batch_sizes(self, batch_sizes: list, num_iterations: int = 50):
        """
        测试不同 batch size 下的性能
        
        Args:
            batch_sizes: batch size 列表
            num_iterations: 每个 batch size 的测试迭代次数
        """
        print("\nBenchmarking different batch sizes...")
        print("=" * 70)
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            print("-" * 70)
            
            self.batch_size = batch_size
            
            # 创建 dummy 输入
            config = self.extractor.config
            
            rgb_height = config.cropped_height if config.crop_image else config.camera_height
            rgb_width = config.cropped_width if config.crop_image else config.camera_width
            dummy_rgb = torch.randn(
                batch_size, 3, rgb_height, rgb_width,
                dtype=torch.float32,
                device=self.device
            ) * 255
            dummy_rgb = dummy_rgb.clamp(0, 255)
            
            lidar_channels = 2 if config.use_ground_plane else 1
            lidar_channels *= config.lidar_seq_len
            dummy_lidar = torch.randn(
                batch_size, lidar_channels,
                config.lidar_resolution_height,
                config.lidar_resolution_width,
                dtype=torch.float32,
                device=self.device
            )
            
            times = []
            
            with torch.no_grad():
                for i in range(num_iterations):
                    torch.cuda.synchronize(self.device)
                    start_time = time.perf_counter()
                    
                    _ = self.extractor(dummy_rgb, dummy_lidar)
                    
                    torch.cuda.synchronize(self.device)
                    end_time = time.perf_counter()
                    
                    times.append((end_time - start_time) * 1000)
            
            times_np = np.array(times)
            avg_time = times_np.mean()
            throughput = batch_size / (avg_time / 1000)
            
            results[batch_size] = {
                'mean_time_ms': avg_time,
                'throughput_samples_per_sec': throughput,
                'times': times_np
            }
            
            print(f"  Mean time: {avg_time:.2f} ms")
            print(f"  Throughput: {throughput:.2f} samples/sec")
        
        # 打印汇总表
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"{'Batch Size':<15} {'Mean Time (ms)':<20} {'Throughput (samples/s)':<25}")
        print("-" * 70)
        for batch_size in batch_sizes:
            mean_time = results[batch_size]['mean_time_ms']
            throughput = results[batch_size]['throughput_samples_per_sec']
            print(f"{batch_size:<15} {mean_time:<20.2f} {throughput:<25.2f}")
        print("=" * 70)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TransFuser Backbone Inference Speed"
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default='/media/z/data/models/garage2/pretrained_models/all_towns',
        help='Path to TransFuser config directory'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--num_iterations',
        type=int,
        default=100,
        help='Number of iterations for benchmark'
    )
    parser.add_argument(
        '--warmup_iterations',
        type=int,
        default=10,
        help='Number of warmup iterations'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use for inference'
    )
    parser.add_argument(
        '--benchmark_batch_sizes',
        action='store_true',
        help='Benchmark different batch sizes'
    )
    
    args = parser.parse_args()
    
    # 创建测试器
    benchmark = InferenceBenchmark(
        config_path=args.config_path,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # 运行测试
    if args.benchmark_batch_sizes:
        # 测试不同 batch size
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        benchmark.benchmark_different_batch_sizes(batch_sizes, num_iterations=50)
    else:
        # 测试单个 batch size
        benchmark.benchmark(
            num_iterations=args.num_iterations,
            warmup_iterations=args.warmup_iterations
        )


if __name__ == "__main__":
    main()
