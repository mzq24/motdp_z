继续上个 session 的工作。先读 `docs/tmp/session_findings_0323.md` 了解完整上下文。

核心发现：训练 validation 函数用 M=1 forward 但模型以 M=34 unified 训练，导致 val 指标不可信（报 0.65 实际 0.40）。

本次需要按顺序做三件事：

1. **修 validation 函数** (`training/train_carla_bev.py` 的 `validate_model`)：当前调用 `compute_loss()` -> `compute_diffusion_loss()` 走 legacy M=1 path。需要改成走 unified forward 或者让 DDIM 推理正确工作。注意 `predict_action` 的 DDIM 推理路径是对的（用 conditional_sample），问题只在 compute_loss 的 reg_loss 指标。

2. **修 checkpoint loading**：两个坑：(a) `register_buffer('name', None)` 的 buffer 不进 state_dict，load_state_dict 会跳过 checkpoint 中的 buffer（如 abs_mean, anchor_centers_abs）；(b) EMA 用 diffusers 格式（shadow_params flat list），不能直接 load_state_dict。见 `scripts/eval_l2_on_dataset.py` 中的临时 workaround。

3. **测试 global abs z-score**：代码已就绪（`global_abs_stats_path` in config），stats 已算好。只需 kill 当前训练 -> 启动新训练。上个 session 实现了三种归一化的 dispatch（global_abs > per-step abs > delta），通过 config 切换。
