# DataLoader Collate 错误调试记录

**日期**: 2026-02-28  
**环境**: NSCC HPC, 4x GPU (torchrun), PyTorch DDP  
**症状**: 训练在 Epoch 1 第 35 个 batch 时必定崩溃

---

## 问题 1: `debug_collate` 不生效

### 现象
自定义的 `debug_collate` 函数没有打印任何调试信息，报错信息仍然是原始的 PyTorch 错误。

### 原因
`debug_collate` 定义在 `world_size == 1`（单 GPU）分支内部，而训练实际使用 `torchrun` 多 GPU 模式（`world_size > 1`），走了另一个分支，使用的是默认 `default_collate`。

### 修复
将 `debug_collate` 定义提到 if/else 之前，并在**两个分支**的 DataLoader 中都传入 `collate_fn=debug_collate`。

---

## 问题 2: `Trying to resize storage that is not resizable`

### 现象
```
RuntimeError: Trying to resize storage that is not resizable
```
每次都在同一个 batch（Epoch 0, Rank 2, Batch 35）崩溃。

### 原因
两个因素叠加：

1. **Storage 不可 resize**: `torch.from_numpy()` 创建的 tensor 与 numpy 数组共享内存，其 storage 不可 resize。当 `num_workers > 0` 时，DataLoader worker 进程尝试将多个 tensor stack 到共享内存中，需要 resize storage，导致崩溃。

2. **Shape 不一致**: `target_point_hist` 字段在不同 sample 中有两种 shape：
   - Old HPC 格式: `(4, 4)` — target_point 和 target_point_next 拼接在一起
   - 标准格式: `(4, 2)` — 仅 target_point

### 为什么每次都是第 35 个 batch？
`DistributedSampler(shuffle=True)` 的 shuffle 是**确定性的**——使用 `epoch` 作为随机种子。对于同一个 epoch（每次都在 epoch 0 崩溃），打乱顺序完全一样，那个 shape 不一致的 sample 每次都被分配到 rank 2 的第 35 个 batch。

### 修复
1. 在 `debug_collate` 中对所有 tensor 执行 `.clone()`，使其拥有独立的可 resize storage
2. 在数据加载时统一 `target_point_hist` 为 `(T, 2)` shape：
   ```python
   tp = torch.from_numpy(value).float()
   final_sample['target_point_hist'] = tp[..., :2]  # 统一截取前2列
   if tp.shape[-1] == 4:
       final_sample['target_point_next_hist'] = tp[..., 2:]  # 拆分出 next
   ```

---

## 问题 3: `KeyError: 'target_point_next_hist'`

### 现象
```
KeyError: Caught KeyError in DataLoader worker process 3.
KeyError: 'target_point_next_hist'
```

### 原因
`default_collate` 要求 batch 内所有 sample 的 key 集合一致。但不同格式的数据 key 不同：

| 数据来源 | `target_point_hist` | `target_point_next_hist` key |
|---|---|---|
| Old HPC packed (`(T,4)`) | 存在 | 从拆分产生 ✓ |
| 标准 preprocess_pdm_lite | `(T,2)` | 独立 key 存在 ✓ |
| 旧版预处理数据 | `(T,2)` | **不存在** ✗ |

当一个 batch 中混合了有/无 `target_point_next_hist` key 的 sample 时，collate 报 KeyError。

### 修复
1. 为 `target_point_next_hist` 添加专门的加载分支（避免被通用 ndarray 分支处理后被覆盖）
2. 在数据转换循环后增加安全检查：缺失时用 `target_point_hist` 的值填充（比补零更合理）
3. 添加计数统计，在每个 epoch 结束时打印缺失数量

### 统计结果
112 batch × 64 batch_size = 7168 samples 中仅 3 个缺失（0.04%），用 target_point 填充完全可以接受。

---

## 最终修改文件

- `dataset/unified_carla_dataset.py`
  - 统一 `target_point_hist` 为 `(T, 2)` shape
  - 显式处理 `target_point_next_hist` 加载
  - 缺失时用 `target_point_hist` 值填充 + warning 日志
- `training/train_carla_bev.py`
  - `debug_collate` → `safe_collate`：精简诊断逻辑，写入 `/tmp` 文件保证 DDP 下可见
  - Epoch 结束打印缺失字段统计
  - **新增 AMP (FP16 混合精度)**：读取 config 中 `model_optimization.use_mixed_precision`，
    使用 `torch.amp.autocast` + `GradScaler` 包裹训练和验证的 forward pass，
    显存约减半，可支持更大 batch_size

---

## 经验总结

1. **多 GPU 训练的 collate_fn 必须在所有代码路径中生效**，不要把自定义 collate 放在单 GPU 分支里 debug
2. **`torch.from_numpy()` 的 tensor 在多 worker DataLoader 中需要 `.clone()`**，否则共享 storage 不可 resize
3. **混合数据集必须保证所有 sample 的 key 集合和 tensor shape 完全一致**，否则 `default_collate` 会崩溃
4. **`DistributedSampler` 的 shuffle 是确定性的（seed = epoch）**，所以同一个 epoch 的崩溃位置固定，不代表没有 shuffle
5. **第一个 epoch 慢（~1h）后续 epoch 快（~5min）是 Linux 页缓存效果**，不是存储系统迁移数据
