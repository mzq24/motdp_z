# Session Findings 2026-03-23 Part 2: M=1 vs M=34 Inference Gap Root Cause

## Key Discovery: Route RoPE Position is the Root Cause

### Problem
M=1 inference gives much worse L2 than M=34 (constructed with dummy anchors):

| Mode | train L2_1s | val L2_1s |
|------|-------------|-----------|
| M=1 (no context) | 0.552 | 0.578 |
| M=34 pos33 (anchor ctx) | 0.092 | 0.168 |
| M=34 pos0 (anchor ctx) | 0.091 | 0.168 |
| M=34 zeros (all zero ctx) | 0.092 | 0.168 |

### Root Cause Analysis

1. **Anchor content doesn't matter**: M=34 with real anchors, zeros, pos0, pos33 all give identical results
2. **The difference is purely structural**: M changes the total sequence length T, which changes route tokens' RoPE positions

#### Code Path (transformer_for_diffusion_multi_head.py)

Decoder input: `x = cat([traj_tokens, route_tokens], dim=1)` → shape `(B, T_traj + T_route, d_model)`
- M=1: T = 1 + 20 = 21, route tokens at positions 1-20
- M=34: T = 34 + 20 = 54, route tokens at positions 34-53

RoPE is applied to ALL tokens based on position in this concatenated sequence:
```python
cos_main, sin_main = self._get_rope_embed(T, ...)  # T=21 vs T=54
q_base = apply_rope_single(q_base, cos_main, sin_main)  # different RoPE for route tokens
```

This makes `route_out` different between M=1 and M=34.

#### Why route_out matters for trajectory
`TrajectoryMLPHead` (line 753) has **route guidance cross-attention**:
```python
self.route_guidance_attn = nn.MultiheadAttention(...)  # traj attends to route
```
Trajectory prediction cross-attends route features in the output head. Different route_out → different trajectory prediction.

#### Why block diagonal mask doesn't help
- Block diagonal mask correctly isolates traj-traj self-attention
- For each traj token, valid KV = 1 self + 64 BEV = 65 (same regardless of M)
- traj_can_attend_route=False blocks traj→route in self-attention
- BUT route tokens themselves get different RoPE → different route_out → different trajectory_head output via route guidance

### What Changed in Code (current state)

1. **Training layout**: `[anchors(0-31), GT(32), x_t(33)]` — x_t at last position (original design)
2. **Model mode_query order**: `[mode_queries, gt_mode_query, diff_mode_query]` (matches training layout)
3. **Inference (conditional_sample)**: constructs M=34 forward with `[anchor_ctx, dummy_gt, x_t]`, takes last slot output
   - `use_m34_inference` flag (default True): enables M=34 constructed inference
   - `m34_slot_order`: 'last' (default) or 'first' — position of x_t (doesn't matter, both give same result)
   - `use_zero_context`: if True, fills anchor slots with zeros instead of real anchors (gives same result)
4. **Eval script** (`scripts/eval_l2_on_dataset.py`): tests 4 modes — M=1, M=34_pos33, M=34_pos0, M=34_zeros

### Experiments Done
- pos=0 training (x_t at first position): worse than pos=33, L2_1s only reached 0.276 after 300 epochs
- pos=33 training (x_t at last position): same result 0.276 — pos doesn't matter
- M=34 constructed inference on both: both reach 0.168 val L2_1s
- All zero context M=34: same 0.168 — content doesn't matter, only M matters

### Next Steps (not yet done)
1. **Fix the root cause**: Make M=1 inference produce same route_out as M=34
   - Option A: Separate RoPE for traj and route segments (route always gets pos 0-19 regardless of M)
   - Option B: Always pad traj to M=34 at inference (current workaround, works but wasteful)
   - Option C: Remove route guidance from trajectory_head (loses useful info)
2. **Training val also needs M=34 inference**: current training val reports M=1 metrics (0.276), need to use constructed M=34
3. **Commit and deploy to HPC**: code changes not yet committed

### Files Modified (uncommitted)
- `policy/annealed_energy_guidance_policy.py` — training layout, conditional_sample with M=34 inference
- `model/transformer_for_diffusion_multi_head.py` — mode_query order
- `scripts/eval_l2_on_dataset.py` — 4-mode eval
- `config/pdm_local_route_b.yaml` — abs_stats_path active
