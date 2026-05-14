# Direct-Prev-Modulated + Shift Sampler Implementation Notes

Date: 2026-05-11

## Context

This note records the implementation for the state next-token collapse fix.

The close-loop issue was that full previous semantic tokens could dominate
current observation and create route-level mode collapse.  The new design keeps
previous semantic state as temporal memory, but prevents it from becoming a
direct copy path.

## Implemented Changes

### 1. Predictor mode cleanup

Implemented in:

- `policy/annealed_energy_guidance_policy.py`
- `config/pdm_hpc_route_b_lidar_bev_stage1_tempocc_0429.yaml`
- `config/pdm_local_route_b_lidar_bev_stage1_tempocc_0429.yaml`

Supported modes now include:

```yaml
semantic_state_predictor_mode: direct_only
semantic_state_predictor_mode: direct_plus_transition
semantic_state_predictor_mode: transition_only
semantic_state_predictor_mode: direct_prev_modulated
```

Backward compatibility:

```text
direct_transition -> direct_plus_transition
```

The current experiment config uses:

```yaml
semantic_state_predictor_mode: direct_prev_modulated
```

### 2. Direct-prev-modulated semantic path

Implemented in:

- `model/transformer_for_diffusion_multi_head.py`
- `policy/annealed_energy_guidance_policy.py`

New model entrypoint:

```python
compute_shared_stage1_prev_modulated_from_ego_outputs(...)
```

Behavior:

```text
current route/context -> direct semantic feature
prev semantic state   -> gamma / beta / route_delta
modulated feature     -> direct_feature * (1 + scale * tanh(gamma)) + scale * beta
modulated route_out   -> route_out + scale * tanh(route_delta)
```

Important guard:

```text
scale = semantic_prev_modulation_scale * prev_valid
```

So when `prev_valid=0`, route start or cache reset falls back to direct
current-frame prediction.

Current config:

```yaml
semantic_prev_modulation_scale: 0.2
semantic_prev_modulation_dropout_prob: 0.5
```

### 3. Fusion removed for direct-prev-modulated

Implemented in:

- `policy/annealed_energy_guidance_policy.py`
- config files

For `direct_prev_modulated`, inference now directly uses the modulated state as
the main semantic state.  It does not fuse direct and transition logits.

Current config:

```yaml
use_semantic_state_fusion: false
semantic_state_fusion_alpha: 0.0
semantic_state_fusion_warmup_frames: 0
semantic_state_fusion_update_cache: true
```

The cache is still updated because the next frame needs predicted previous
state for modulation.

### 4. Prev-state corruption and dropout

Implemented in:

- `policy/annealed_energy_guidance_policy.py`

Function updated:

```python
_apply_semantic_transition_prev_dropout(...)
```

It now supports:

- sample-level prev reset dropout
- group-level dropout for window / dir / area / tempocc / phase / graph / boundary / chase
- categorical random replacement
- categorical replace-to-none
- optional prefix dropout if route-local frame index exists

Current config:

```yaml
semantic_transition_prev_dropout_prob: 0.2
semantic_prev_token_dropout_prob: 0.6
semantic_prev_window_dropout_prob: 0.9
semantic_prev_dir_dropout_prob: 0.9
semantic_prev_area_dropout_prob: 0.8
semantic_prev_tempocc_dropout_prob: 0.8
semantic_prev_phase_dropout_prob: 0.7
semantic_prev_graph_dropout_prob: 0.7
semantic_prev_boundary_dropout_prob: 0.5
semantic_prev_chase_dropout_prob: 0.5
semantic_prev_random_replace_prob: 0.2
semantic_prev_replace_to_none_prob: 0.5
semantic_prev_prefix_dropout_frames: 8
semantic_prev_prefix_dropout_prob: 1.0
```

Prefix dropout is best-effort.  It only activates if the batch contains a
route-local frame index such as:

```text
semantic_route_frame_index
semantic_prev_route_frame_index
route_local_frame_index
route_frame_index
frame_in_route
sample_route_index
```

If none exist, training continues with normal corruption only.

### 5. Semantic shift-aware sampler

Implemented in:

- `training/train_carla_bev.py`
- HPC/local config files

The existing window-aware weighted sampler is preserved.  Semantic shift bonus
is added on top of the existing weight and clamped.

Current HPC config:

```yaml
use_semantic_shift_sampler: true
semantic_shift_window_weight: 4.0
semantic_shift_phase_weight: 3.0
semantic_shift_area_weight: 2.0
semantic_shift_edge_weight: 2.0
semantic_shift_opportunity_weight: 2.0
semantic_shift_max_weight: 8.0
```

Shift signals:

- window/family changes between `prev_conflict_area_family` and current
  `conflict_area_family`
- decision/control phase changes
- area status changes
- current/future edge valid or mode changes, only if prev edge fields exist
- `go_opportunity_prob` or `yld_pressure_prob` crossing `0.5`

Boundary scalar shifts are intentionally not used as sampler signals in this
version to avoid oversampling label jitter.

### 6. Explicit route-intent token residual

Implemented in:

- `model/transformer_for_diffusion_multi_head.py`
- `policy/annealed_energy_guidance_policy.py`
- HPC/local config files

Motivation:

```text
target_point / target_point_next already entered ego_status, but route saw them
only through a mixed conditioning vector that can be group-dropped.  Exit/ramp
branching needs target intent to be explicit enough for route selection, while
still not becoming a hard planner override.
```

New config:

```yaml
use_route_intent_token: true
route_intent_gate_init: 0.1
```

Input layout:

```text
command(6) + target_point(2) + target_point_next(2)
```

Implementation:

```text
route_intent_emb = MLP(current_status[:, 2:12])
route_emb += sigmoid(route_intent_gate) * route_intent_emb
```

This uses the existing condition-group dropout path, so target intent dropout is
still preserved.  The small gate makes target intent audible to route tokens
without forcing route to blindly follow the planner.

## Debug / Outputs

Policy output now includes:

```text
semantic_state_predictor_mode_resolved
semantic_prev_corruption_enabled
semantic_state_fusion_enabled
semantic_state_fusion_gate
```

For `direct_prev_modulated`, fusion should be disabled in config, so fusion
debug is only compatibility/debug information.

## Validation

Static checks passed:

```bash
python -m py_compile \
  policy/annealed_energy_guidance_policy.py \
  model/transformer_for_diffusion_multi_head.py \
  training/train_carla_bev.py \
  dataset/unified_carla_dataset.py

/home/z/anaconda3/envs/dpauto/bin/python -c "import yaml; ..."

git diff --check
```

## Notes / Risks

- `semantic_prev_prefix_dropout_*` may be inactive unless the packed dataset
  exposes route-local frame index fields.
- `direct_prev_modulated` still updates predicted semantic cache; this is
  required for next-frame modulation.
- `direct_plus_transition` and `transition_only` remain available for ablation,
  but the current experiment config defaults to `direct_prev_modulated`.
