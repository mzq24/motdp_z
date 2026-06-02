# Continual Validation / Backfill Notes

## Why backfill is needed

Current continual-learning training writes `continual_validation.json` with `validation_mode: seen_targets`.
That file is useful for online training sanity checks, but it is **not** a full all-target / future-task matrix:

- after task 1, it validates only task 1;
- after task 2, it validates only tasks 1-2;
- ...
- after final task, it validates all seen tasks.

So it can show current-task learning and final forgetting on seen tasks, but it cannot show how a checkpoint after task `i` performs on future task `j > i`.

For future-task / full-target analysis, evaluate every task-end checkpoint on **all target tasks** after training.

## Reporting protocol

Use the presentation-style CL retention matrix by default:

- rows: eval target task `Tj`;
- columns: checkpoints after training `T1..Ti`;
- diagonal: current-task performance, measured right after the task is trained;
- upper triangle: retained performance of old tasks after later tasks are trained;
- blank lower triangle: task has not been trained yet at that checkpoint.

Example shape:

| Eval task | after T1 | after T2 | after T3 |
| --- | ---: | ---: | ---: |
| T1 | diag | retention | retention |
| T2 |  | diag | retention |
| T3 |  |  | diag |

If a script outputs rows as `after Ti` and columns as eval task `Tj`, the same data appears as a lower-triangle prefix table. For presentation and discussion, transpose it back to the canonical retention upper triangle.

Matrix names:

- **retention upper triangle**: the canonical CL table above. This is the main table for forgetting, retention, and presentation.
- **future-task matrix**: evaluate `after Ti` checkpoints on tasks `Tj > i`, before those tasks are trained. This is useful for task similarity and recovery/order analysis, but it is not the standard CL retention table.
- **full all-target matrix**: evaluate every `after Ti` checkpoint on all target tasks. This contains both retention and future-task views and is safest for post-hoc analysis.

Online `seen_targets` validation is enough for the retention upper triangle if it evaluates all seen tasks after every task. Full all-target backfill is required for future-task analysis or when online validation missed cells.

## CL7 LoRA backfill command

Script:

```bash
cd /data/z_project/code/nuplan_whitenoise_diffusion_v1
bash tmp/run_lora_upper_triangle_backfill_20260602.sh
```

This runs:

- Normal LoRA rank16 checkpoints on GPUs 0-3;
- PEGP LoRA rank16 checkpoints on GPUs 4-7;
- each task-end checkpoint is evaluated on all 7 CL7 target tasks;
- each per-task validation uses `--max-batches 10`, `--val-batch-size 64`, `--num-workers 4`.

Output root:

```text
/workspace2/z_project/exp/nuplan/nuplan_whitenoise_diffusion_v1/backfill_lora_cl7_upper_triangle_20260602/
```

Important outputs:

```text
normal/upper_triangle_validation.json
normal/upper_triangle_validation.md
pegp/upper_triangle_validation.json
pegp/upper_triangle_validation.md
lora_upper_triangle_comparison.md
logs/runner.log
```

## Interpretation

Rows are task-end checkpoints: `after T1`, `after T2`, ... `after T7`.
Columns are all target tasks, including future tasks. Lower ADE/FDE is better.

Use the backfilled matrix for:

- upper-triangle / future-task behavior;
- recovery vs monotonic forgetting diagnosis;
- normal vs PEGP comparison at matched checkpoints.

Use training-time `continual_validation.json` only for:

- online sanity checks;
- seen-target curves;
- final all-seen-task metrics.
