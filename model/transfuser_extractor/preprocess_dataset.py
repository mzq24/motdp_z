"""
数据集预处理脚本
================
两阶段流水线，解决 Lustre MDS jitter（大量小文件 open/stat 导致的偶发延迟）：

  Phase 1 (pack_source):
    读 .laz + .jpg → 预处理 → 打包为 route_source.pt（per-route 大文件）
    每个 route 一个文件，彻底消除 Phase 2 的小文件 IO。
    可在 CPU 节点上独立完成，之后无需重跑。

  Phase 2 (extract):
    读 route_source.pt（若存在）→ GPU TransFuser → 保存 route_features.pt
    Route 级别预取：IO 线程加载 route[i+1] 时 GPU 处理 route[i]。
    若 route_source.pt 不存在，回退到原有的 per-frame 并行 IO + 预取路径。

使用方法:
    # 全流程（先 pack 源数据，再 GPU 提取）
    python preprocess_dataset.py --mode pack_and_extract ...

    # 仅 pack 源数据（CPU 节点）
    python preprocess_dataset.py --mode pack_source ...

    # 仅 GPU 提取（源数据已 pack）
    python preprocess_dataset.py --mode extract ...

route_source.pt 格式:
    {
        'frame_nums': List[str],                  # ['0001', '0002', ...]
        'rgbs':       Tensor (N, 3, H, W) uint8,  # 裁剪后的 RGB，节省约 75% 空间
        'lidar_bevs': Tensor (N, C, 256, 256) float16,
    }

route_features.pt 格式:
    {
        'frame_nums':  List[str],
        'bev_features':  Tensor (N, 1512, 8, 8),
        'bev_upsamples': Tensor (N, 64, 64, 64),
    }
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
import cv2
import laspy

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from backbone_extractor import TransFuserBackboneExtractor
import transfuser_utils as t_u


class DatasetPreprocessor:
    """
    数据集预处理器

    遍历 pdm_lite 数据集，提取并保存 TransFuser BEV 特征。
    支持两阶段流水线以规避 Lustre MDS 小文件瓶颈。
    """

    def __init__(self,
                 dataset_path: str,
                 config_path: str,
                 model_path: str = None,
                 device: str = 'cuda:0',
                 batch_size: int = 1,
                 skip_existing: bool = True):
        self.dataset_path = Path(dataset_path)
        self.config_path = config_path
        self.device = device
        self.batch_size = batch_size
        self.skip_existing = skip_existing

        print("Initializing TransFuser Backbone Extractor...")
        self.extractor = TransFuserBackboneExtractor(
            config_path=config_path,
            model_path=model_path,
            device=device
        )
        self.config = self.extractor.config

        self.feature_dir_name = "transfuser_feature"

    # ------------------------------------------------------------------
    # 数据集发现
    # ------------------------------------------------------------------

    def find_all_routes(self):
        """
        查找数据集中所有有效的 route 目录。

        Lustre 优化：
        - os.scandir 复用 readdir 返回的 d_type，避免每次 is_dir() 额外 stat
        - 去掉 gzip.open 验证（仅做 exists 检查），消除 2500 次文件读取
        - ThreadPoolExecutor 并行化 per-route 的 exists 检查
        """
        import os
        from concurrent.futures import ThreadPoolExecutor

        def _check_route(route_path: str):
            p = Path(route_path)
            if p.name.startswith('FAILED_'):
                return None
            if ((p / 'lidar').exists() and
                    (p / 'rgb').exists() and
                    (p / 'results.json.gz').exists()):
                return p
            return None

        # 收集候选 route 路径（两层 scandir，利用 d_type 跳过非目录 stat）
        candidates = []
        try:
            with os.scandir(self.dataset_path) as sit:
                for scenario_entry in sit:
                    if not scenario_entry.is_dir(follow_symlinks=False):
                        continue
                    try:
                        with os.scandir(scenario_entry.path) as rit:
                            for route_entry in rit:
                                if route_entry.is_dir(follow_symlinks=False):
                                    candidates.append(route_entry.path)
                    except PermissionError:
                        continue
        except PermissionError:
            pass

        # 并行 exists 检查（32 线程，每线程独立 stat）
        with ThreadPoolExecutor(max_workers=32) as executor:
            results = list(executor.map(_check_route, candidates))

        return sorted(p for p in results if p is not None)

    def get_frame_count(self, route_dir: Path) -> int:
        return len(list((route_dir / 'lidar').glob('*.laz')))

    # ------------------------------------------------------------------
    # 单帧预处理（线程安全）
    # ------------------------------------------------------------------

    def preprocess_rgb(self, rgb_path: str) -> torch.Tensor:
        """读取并裁剪 RGB 图像，返回 (1, 3, H, W) float32 [0, 255]"""
        image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = t_u.crop_array(self.config, image)
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image).float().unsqueeze(0)

    def preprocess_lidar(self, lidar_path: str) -> torch.Tensor:
        """读取 .laz 并转换为 BEV histogram，返回 (1, C, 256, 256) float32"""
        las_object = laspy.read(lidar_path)
        lidar = las_object.xyz
        lidar_bev = self.extractor.lidar_to_histogram_features(
            lidar, use_ground_plane=self.config.use_ground_plane)
        return torch.from_numpy(lidar_bev).float().unsqueeze(0)

    def _load_frame(self, args):
        """线程安全的单帧读取，供 ThreadPoolExecutor 调用"""
        lidar_file, rgb_file, frame_num = args
        try:
            rgb = self.preprocess_rgb(str(rgb_file))
            lidar_bev = self.preprocess_lidar(str(lidar_file))
            return frame_num, rgb, lidar_bev
        except Exception as e:
            print(f"Error reading frame {frame_num}: {e}")
            return frame_num, None, None

    # ------------------------------------------------------------------
    # Phase 1: 打包源数据
    # ------------------------------------------------------------------

    def pack_source_route(self, route_dir: Path):
        """
        将 route 下所有 .laz + .jpg 预处理并打包为 route_source.pt。

        存储格式：
          rgbs:       (N, 3, H, W) uint8   — 比 float32 节省 75% 空间
          lidar_bevs: (N, C, 256, 256) float16 — 比 float32 节省 50% 空间

        所有帧并行读取（无 GPU），最多 32 个 IO 线程。
        """
        from concurrent.futures import ThreadPoolExecutor

        source_pack_path = route_dir / 'route_source.pt'
        if self.skip_existing and source_pack_path.exists():
            return

        lidar_dir = route_dir / 'lidar'
        rgb_dir = route_dir / 'rgb'
        frame_files = sorted(lidar_dir.glob('*.laz'))

        pending = []
        for lidar_file in frame_files:
            frame_num = lidar_file.stem
            rgb_file = rgb_dir / f"{frame_num}.jpg"
            if not rgb_file.exists():
                print(f"Warning: RGB file not found: {rgb_file}")
                continue
            pending.append((lidar_file, rgb_file, frame_num))

        if not pending:
            return

        # 全部帧并行读取（IO bound，无 GPU）
        io_workers = min(len(pending), 32)
        results = {}

        with ThreadPoolExecutor(max_workers=io_workers) as executor:
            futures = [executor.submit(self._load_frame, args) for args in pending]
            for future in tqdm(futures, desc="  reading frames", unit="frame", leave=False):
                fn, rgb, lidar_bev = future.result()
                if rgb is not None:
                    # squeeze batch dim，存为紧凑格式
                    results[fn] = (rgb.squeeze(0).to(torch.uint8),   # (3, H, W) uint8
                                   lidar_bev.squeeze(0).half())       # (C, 256, 256) float16

        if not results:
            return

        sorted_fns = sorted(results.keys())
        rgbs = torch.stack([results[fn][0] for fn in sorted_fns])       # (N, 3, H, W) uint8
        lidar_bevs = torch.stack([results[fn][1] for fn in sorted_fns]) # (N, C, 256, 256) float16

        torch.save({
            'frame_nums': sorted_fns,
            'rgbs': rgbs,
            'lidar_bevs': lidar_bevs,
        }, source_pack_path)

    # ------------------------------------------------------------------
    # Phase 2: 提取 TransFuser features
    # ------------------------------------------------------------------

    def process_route(self, route_dir: Path, source_pack: dict = None):
        """
        提取单个 route 的 BEV feature，保存为 route_features.pt。

        source_pack 传入时走快速路径（无 IO，直接从内存 tensor 批量推理）。
        source_pack 为 None 时回退到 per-frame 并行 IO + batch 预取路径。
        """
        feature_dir = route_dir / self.feature_dir_name
        feature_dir.mkdir(exist_ok=True)

        packed_path = feature_dir / 'route_features.pt'
        if self.skip_existing and packed_path.exists():
            return

        all_frame_nums = []
        all_bev_features = []
        all_bev_upsamples = []

        if source_pack is not None:
            # ---- 快速路径：源数据已在内存，无文件 IO ----
            frame_nums = source_pack['frame_nums']
            rgbs = source_pack['rgbs'].float()        # uint8 → float32 [0, 255]
            lidar_bevs = source_pack['lidar_bevs'].float()  # float16 → float32

            n_batches = (len(frame_nums) + self.batch_size - 1) // self.batch_size
            for i in tqdm(range(0, len(frame_nums), self.batch_size),
                          desc="  GPU batches", unit="batch", total=n_batches, leave=False):
                batch_frame_nums = frame_nums[i:i + self.batch_size]
                rgb_batch = rgbs[i:i + self.batch_size]
                lidar_batch = lidar_bevs[i:i + self.batch_size]
                try:
                    with torch.no_grad():
                        output = self.extractor(rgb_batch, lidar_batch)
                    all_frame_nums.extend(batch_frame_nums)
                    all_bev_features.append(output['bev_feature'].cpu())
                    all_bev_upsamples.append(output['bev_feature_upscale'].cpu())
                except Exception as e:
                    print(f"Error processing batch in {route_dir}: {e}")
                    continue

        else:
            # ---- 慢速回退路径：per-frame 并行 IO + batch 预取 ----
            from concurrent.futures import ThreadPoolExecutor

            lidar_dir = route_dir / 'lidar'
            rgb_dir = route_dir / 'rgb'
            frame_files = sorted(lidar_dir.glob('*.laz'))

            pending = []
            for lidar_file in frame_files:
                frame_num = lidar_file.stem
                rgb_file = rgb_dir / f"{frame_num}.jpg"
                if not rgb_file.exists():
                    print(f"Warning: RGB file not found: {rgb_file}")
                    continue
                pending.append((lidar_file, rgb_file, frame_num))

            if not pending:
                return

            batches = [pending[i:i + self.batch_size]
                       for i in range(0, len(pending), self.batch_size)]

            io_workers = min(self.batch_size, 32)

            with ThreadPoolExecutor(max_workers=io_workers) as executor:
                def _submit_batch(batch_args):
                    return {executor.submit(self._load_frame, a): a[2] for a in batch_args}

                pending_futures = _submit_batch(batches[0])

                for i, batch_args in tqdm(enumerate(batches), desc="  GPU batches",
                                          unit="batch", total=len(batches), leave=False):
                    loaded = {}
                    for future in pending_futures:
                        fn, rgb, lidar_bev = future.result()
                        if rgb is not None:
                            loaded[fn] = (rgb, lidar_bev)

                    if i + 1 < len(batches):
                        pending_futures = _submit_batch(batches[i + 1])

                    if not loaded:
                        continue

                    ordered = [(fn, loaded[fn]) for _, _, fn in batch_args if fn in loaded]
                    batch_frame_nums = [fn for fn, _ in ordered]
                    rgb_batch = torch.cat([d[0] for _, d in ordered], dim=0)
                    lidar_batch = torch.cat([d[1] for _, d in ordered], dim=0)

                    try:
                        with torch.no_grad():
                            output = self.extractor(rgb_batch, lidar_batch)
                        all_frame_nums.extend(batch_frame_nums)
                        all_bev_features.append(output['bev_feature'].cpu())
                        all_bev_upsamples.append(output['bev_feature_upscale'].cpu())
                    except Exception as e:
                        print(f"Error processing batch in {route_dir}: {e}")
                        continue

        if not all_bev_features:
            return

        torch.save({
            'frame_nums': all_frame_nums,
            'bev_features': torch.cat(all_bev_features, dim=0),   # (N, 1512, 8, 8)
            'bev_upsamples': torch.cat(all_bev_upsamples, dim=0),  # (N, 64, 64, 64)
        }, packed_path)

    # ------------------------------------------------------------------
    # 提取循环（带 route 级别预取）
    # ------------------------------------------------------------------

    def _run_extract(self, routes: list):
        """
        带 route 级别预取的特征提取主循环。

        IO 线程在后台加载 route[i+1] 的 route_source.pt，
        同时 GPU 处理 route[i] 的推理。两者充分重叠。
        """
        from concurrent.futures import ThreadPoolExecutor

        def _load_source_pack(route_dir: Path):
            path = route_dir / 'route_source.pt'
            if path.exists():
                return torch.load(path, weights_only=True)
            return None  # 触发慢速回退路径

        with ThreadPoolExecutor(max_workers=1) as io_executor:
            # 预提交第一个 route 的 IO
            next_future = io_executor.submit(_load_source_pack, routes[0])

            for i, route_dir in enumerate(tqdm(routes, desc="Extracting")):
                source_pack = next_future.result()  # 等待当前 route 的 IO 完成

                # 立即提交下一个 route 的 IO（与当前 GPU 推理并行）
                if i + 1 < len(routes):
                    next_future = io_executor.submit(_load_source_pack, routes[i + 1])

                frame_count = self.get_frame_count(route_dir)
                tqdm.write(f"  {route_dir.name} ({frame_count} frames)"
                           f"{'  [source pack]' if source_pack is not None else '  [fallback IO]'}")

                self.process_route(route_dir, source_pack=source_pack)

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def run(self, mode: str = 'extract'):
        """
        运行预处理流水线。

        Args:
            mode: 运行模式
                'pack_source'      — 仅 Phase 1：打包源数据（CPU 节点可用）
                'extract'          — 仅 Phase 2：提取 BEV feature（需 GPU）
                'pack_and_extract' — Phase 1 + Phase 2 连续执行
        """
        print("\n" + "=" * 60)
        print(f"Dataset Preprocessing  [mode={mode}]")
        print("=" * 60)
        print(f"Dataset path:  {self.dataset_path}")
        print(f"Skip existing: {self.skip_existing}")

        print("\nFinding all routes...")
        routes = self.find_all_routes()
        print(f"Found {len(routes)} valid routes")

        if not routes:
            print("No valid routes found. Exiting.")
            return

        total_frames = sum(self.get_frame_count(r) for r in routes)
        print(f"Total frames:  {total_frames}")

        # Phase 1
        if mode in ('pack_source', 'pack_and_extract'):
            print("\n[Phase 1] Packing source data → route_source.pt ...")
            for route_dir in tqdm(routes, desc="Packing source"):
                self.pack_source_route(route_dir)
            print("Phase 1 complete.")

        # Phase 2
        if mode in ('extract', 'pack_and_extract'):
            print("\n[Phase 2] Extracting TransFuser features → route_features.pt ...")
            self._run_extract(routes)
            print("Phase 2 complete.")

        print("\n" + "=" * 60)
        print("Done.")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess pdm_lite dataset and extract TransFuser BEV features"
    )
    parser.add_argument('--dataset_path', type=str,
                        default='/home/wang/Dataset/pdm_lite_mini')
    parser.add_argument('--config_path', type=str,
                        default='/home/wang/Project/carla_garage/leaderboard/leaderboard/pretrained_models/all_towns')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model weights (optional, auto-detected from config_path)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--no_skip_existing', action='store_true',
                        help='Do not skip existing output files')
    parser.add_argument('--mode', type=str, default='extract',
                        choices=['pack_source', 'extract', 'pack_and_extract'],
                        help=(
                            'pack_source: Phase 1 only — pack .laz+.jpg into route_source.pt (CPU node); '
                            'extract: Phase 2 only — GPU TransFuser inference (needs route_source.pt or falls back); '
                            'pack_and_extract: run both phases sequentially'
                        ))

    args = parser.parse_args()

    preprocessor = DatasetPreprocessor(
        dataset_path=args.dataset_path,
        config_path=args.config_path,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        skip_existing=not args.no_skip_existing,
    )

    preprocessor.run(mode=args.mode)


if __name__ == "__main__":
    main()
