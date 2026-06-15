# Legacy e60 Nostate Recovery Ladder

## Goal

Recover the strong legacy nostate e60 result from the clean motion-only path
without changing multiple causes at once. Every rung uses four GPUs, seed
`20260614`, the same data split, old pred-x0 DDIM, route intent, GPS noise
`sigma=0.05`, and motion-only losses.

The configured learning rate is `5e-5`. Both clean and legacy trainers apply
the same four-GPU linear scaling, so the effective initial learning rate is
`2e-4`; the final learning rate remains `1e-7`.

## Stages

| Stage | Policy/trainer | Structure delta | State behavior |
|---|---|---|---|
| R0 | clean | N3 unified decoder, condition depth 4 | no state modules |
| R1 | clean | R0 + condition depth 6 | no state modules |
| R2 | legacy | R1 + legacy policy/train wrapper and legacy motion recipe | `motion_only_model=true` |
| R3-natural | legacy | R2 + full state topology | state paths off; semantic params frozen |
| R3-common | legacy | R3-natural + R2 epoch-0 motion initialization | state paths off; semantic params frozen |
| R4 | legacy | legacy-exact full topology and optimizer/DDP behavior | state paths off; semantic params remain in optimizer |

R2, R3 and R4 share the recovered legacy config skeleton. R2 changes only
`motion_only_model`; R3 and R4 construct the full topology. R3 freezes all
semantic-only parameters before optimizer construction, while R4 preserves the
original unused-parameter behavior.

## Strict Common-Init Check

R2 writes:

```text
checkpoints/legacy_e60_recovery_r2_legacy_wrapper_motion_only_0614/initial_model.pt
```

R3-common loads only matching non-semantic tensors from this file. Its
`motion_initialization_sha256` must equal R2. R3-natural deliberately allows
full-state construction to consume random numbers and change later motion-head
initialization.

Each run writes `structure_manifest.json` with config/commit hashes, parameter
counts, flags, effective LR and the motion initialization hash.

Local structure audit on 2026-06-14:

```text
R2 total params:                  46,643,094
R2 semantic params:                       0
R3/R4 total params:              71,352,360
R3/R4 semantic params:           24,709,266
R3 semantic trainable params:             0
R4 semantic trainable params:    24,709,266
```

After loading R2 `motion_common` into R3-common, the motion initialization
hashes match. An eval-mode forward on identical synthetic inputs produced
exactly zero maximum absolute difference for all six motion outputs.

## Config Diff Record

```text
R0 -> R1
  policy.n_cond_layers: 4 -> 6

R1 -> R2
  PaperMotionPolicy/train_motion_only_clean.py
    -> AnnealedEnergyGuidancePolicy/train_carla_bev.py
  model remains structure-level motion-only

R2 -> R3-natural
  route_b.motion_only_model: true -> false
  route_b.freeze_unused_semantic_modules: false -> true

R3-natural -> R3-common
  training.init_scope: motion_common
  training.init_checkpoint: R2/initial_model.pt

R3-natural -> R4
  route_b.freeze_unused_semantic_modules: true -> false
  semantic parameters remain unused but are retained in optimizer/EMA behavior
```

## Commands

Run one stage on four GPUs:

```bash
cd /data/z_project/code/motdp_z_semantic_state_strict_ablation_v1

CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 \
  bash scripts/codex_bash/train_legacy_e60_r0_0614.sh
```

Replace `r0` with `r1`, `r2`, `r3_natural`, `r3_common`, or `r4` for the other
entry points. Run the complete sequence with:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 \
  bash scripts/codex_bash/train_legacy_e60_recovery_ladder_0614.sh
```

`R3-common` requires R2 to have started successfully and written its epoch-0
checkpoint.

## Result Table

| Stage | Commit | Best epoch | L2 avg | Route L2 | Route final | Speed MAE | Close-loop |
|---|---|---:|---:|---:|---:|---:|---:|
| R0 | `ff874cf` | - | - | - | - | - | - |
| R1 | `ef5f851` | - | - | - | - | - | - |
| R2 | `7426d08` | - | - | - | - | - | - |
| R3-natural | `6f6e472` | - | - | - | - | - | - |
| R3-common | `6f6e472` | - | - | - | - | - | - |
| R4 | `db7db9c` | - | - | - | - | - | - |

Only a stage reaching `route_L2 < 0.09` and `route_final < 0.18`, while
improving over the previous rung, is promoted to close-loop evaluation.

## Close-Loop Launchers

All launchers default to four GPUs (`0,1,2,3`), eight tasks, ten inference
steps, and the 218-route `bench2drive220_skip_23695_24071` set. Existing result
shards resume in place through the evaluator's `--resume=True` behavior.

```bash
bash scripts/codex_bash/closeloop_legacy_e60_r0_0615.sh
bash scripts/codex_bash/closeloop_legacy_e60_r1_0615.sh
bash scripts/codex_bash/closeloop_legacy_e60_r2_0615.sh
bash scripts/codex_bash/closeloop_legacy_e60_r3_natural_0615.sh
bash scripts/codex_bash/closeloop_legacy_e60_r3_common_0615.sh
bash scripts/codex_bash/closeloop_legacy_e60_r4_0615.sh
```

Default candidate epochs are R0 e45, R1 e60, R2 e55, R3-natural e55,
R3-common e55, and R4 e55. Override any candidate without editing a file:

```bash
CKPT_EPOCH=60 CUDA_VISIBLE_DEVICES=4,5,6,7 \
  bash scripts/codex_bash/closeloop_legacy_e60_r3_natural_0615.sh
```

## Interpretation

- R1 improvement isolates condition encoder depth.
- R2 improvement isolates the legacy wrapper and its training behavior.
- R3-natural improvement without R3-common improvement indicates a favorable
  initialization basin, not semantic information.
- R3-common improvement despite identical motion initialization requires an
  audit for an unintended state-to-motion path.
- R4 improvement over R3 confirms that retaining unused semantic parameters in
  the original optimizer/DDP/EMA path changes optimization behavior.

## Git Record

| Change | Commit |
|---|---|
| cond6 and legacy-wrapper baseline support | `064126b` |
| deterministic recovery infrastructure / R0 | `ff874cf` |
| R1 config | `ef5f851` |
| R2 config | `7426d08` |
| R3 natural/common controls | `6f6e472` |
| R4 exact config and launch sequence | `db7db9c` |
