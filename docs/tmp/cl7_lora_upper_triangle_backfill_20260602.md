# CL7 LoRA Upper-Triangle Backfill Result

Generated: 2026-06-02 on 40G host.

This records the all-target backfill for the CL7 monotonic LoRA runs. Training-time continual validation was seen-target only; these tables evaluate every task-end checkpoint on all 7 target tasks.

Output root:

```text
/workspace2/z_project/exp/nuplan/nuplan_whitenoise_diffusion_v1/backfill_lora_cl7_upper_triangle_20260602/
```

# CL7 LoRA All-Target Backfill Comparison

Lower ADE/FDE is better. `PEGP - Normal` below: negative means PEGP is better.

## Presentation-Style Retention Upper Triangle: ego_ADE

Rows are eval tasks and columns are task-end checkpoints. This matches the previous presentation format: diagonal cells are current-task performance; upper-triangle cells are old-task retention after later finetuning. Lower ADE is better.

### Normal LoRA r16

| Eval task | after T1 | after T2 | after T3 | after T4 | after T5 | after T6 | after T7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T1 traffic | 4.555 | 6.319 | 6.488 | 5.563 | 4.925 | 10.554 | 13.379 |
| T2 following |  | 3.452 | 3.979 | 4.782 | 4.315 | 5.608 | 11.919 |
| T3 high_lat |  |  | 5.168 | 5.510 | 5.073 | 8.725 | 11.707 |
| T4 near_multi |  |  |  | 4.907 | 4.324 | 11.425 | 14.626 |
| T5 ped_wait |  |  |  |  | 3.790 | 5.890 | 7.063 |
| T6 pickup |  |  |  |  |  | 6.065 | 7.707 |
| T7 stationary |  |  |  |  |  |  | 2.033 |

### PEGP LoRA r16

| Eval task | after T1 | after T2 | after T3 | after T4 | after T5 | after T6 | after T7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T1 traffic | 4.752 | 6.565 | 6.737 | 5.826 | 5.015 | 10.011 | 11.736 |
| T2 following |  | 3.483 | 4.183 | 5.190 | 3.950 | 5.083 | 11.379 |
| T3 high_lat |  |  | 5.199 | 5.429 | 5.517 | 8.202 | 11.068 |
| T4 near_multi |  |  |  | 5.348 | 4.629 | 11.531 | 13.937 |
| T5 ped_wait |  |  |  |  | 3.682 | 5.575 | 7.076 |
| T6 pickup |  |  |  |  |  | 6.055 | 7.085 |
| T7 stationary |  |  |  |  |  |  | 1.925 |

## Seen-Task Prefix Triangle: ego_ADE

This is the standard CL table intended here: after training tasks `T1..Ti`, evaluate only the seen tasks `T1..Ti`. Lower ADE is better.

### Normal LoRA r16

| after / eval | T1 traffic | T2 following | T3 high_lat | T4 near_multi | T5 ped_wait | T6 pickup | T7 stationary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| after T1 traffic | 4.555 |  |  |  |  |  |  |
| after T2 following | 6.319 | 3.452 |  |  |  |  |  |
| after T3 high_lat | 6.488 | 3.979 | 5.168 |  |  |  |  |
| after T4 near_multi | 5.563 | 4.782 | 5.510 | 4.907 |  |  |  |
| after T5 ped_wait | 4.925 | 4.315 | 5.073 | 4.324 | 3.790 |  |  |
| after T6 pickup | 10.554 | 5.608 | 8.725 | 11.425 | 5.890 | 6.065 |  |
| after T7 stationary | 13.379 | 11.919 | 11.707 | 14.626 | 7.063 | 7.707 | 2.033 |

### PEGP LoRA r16

| after / eval | T1 traffic | T2 following | T3 high_lat | T4 near_multi | T5 ped_wait | T6 pickup | T7 stationary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| after T1 traffic | 4.752 |  |  |  |  |  |  |
| after T2 following | 6.565 | 3.483 |  |  |  |  |  |
| after T3 high_lat | 6.737 | 4.183 | 5.199 |  |  |  |  |
| after T4 near_multi | 5.826 | 5.190 | 5.429 | 5.348 |  |  |  |
| after T5 ped_wait | 5.015 | 3.950 | 5.517 | 4.629 | 3.682 |  |  |
| after T6 pickup | 10.011 | 5.083 | 8.202 | 11.531 | 5.575 | 6.055 |  |
| after T7 stationary | 11.736 | 11.379 | 11.068 | 13.937 | 7.076 | 7.085 | 1.925 |

## Future-Task Triangle With Diagonal: ego_ADE

Rows and columns use the same CL7 target order. The diagonal is current-task performance; the strict upper triangle is future-task performance before the future task is trained. Lower triangle entries are intentionally omitted here.

### Normal LoRA r16

| after / eval | T1 traffic | T2 following | T3 high_lat | T4 near_multi | T5 ped_wait | T6 pickup | T7 stationary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| after T1 traffic | 4.555 | 3.745 | 5.007 | 3.931 | 4.059 | 5.119 | 1.571 |
| after T2 following |  | 3.452 | 5.514 | 5.384 | 4.506 | 5.384 | 2.390 |
| after T3 high_lat |  |  | 5.168 | 5.851 | 4.784 | 5.687 | 2.776 |
| after T4 near_multi |  |  |  | 4.907 | 5.171 | 6.069 | 4.993 |
| after T5 ped_wait |  |  |  |  | 3.790 | 4.867 | 1.932 |
| after T6 pickup |  |  |  |  |  | 6.065 | 2.384 |
| after T7 stationary |  |  |  |  |  |  | 2.033 |

### PEGP LoRA r16

| after / eval | T1 traffic | T2 following | T3 high_lat | T4 near_multi | T5 ped_wait | T6 pickup | T7 stationary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| after T1 traffic | 4.752 | 3.726 | 5.239 | 4.006 | 4.197 | 5.133 | 1.510 |
| after T2 following |  | 3.483 | 5.665 | 5.648 | 4.360 | 5.174 | 2.255 |
| after T3 high_lat |  |  | 5.199 | 6.816 | 4.925 | 5.480 | 2.997 |
| after T4 near_multi |  |  |  | 5.348 | 5.392 | 6.267 | 4.772 |
| after T5 ped_wait |  |  |  |  | 3.682 | 4.897 | 1.949 |
| after T6 pickup |  |  |  |  |  | 6.055 | 2.574 |
| after T7 stationary |  |  |  |  |  |  | 1.925 |

### PEGP - Normal Delta

Negative means PEGP is better.

| after / eval | T1 traffic | T2 following | T3 high_lat | T4 near_multi | T5 ped_wait | T6 pickup | T7 stationary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| after T1 traffic | +0.196 | -0.019 | +0.232 | +0.074 | +0.138 | +0.014 | -0.061 |
| after T2 following |  | +0.031 | +0.151 | +0.264 | -0.147 | -0.209 | -0.135 |
| after T3 high_lat |  |  | +0.031 | +0.965 | +0.141 | -0.207 | +0.221 |
| after T4 near_multi |  |  |  | +0.441 | +0.222 | +0.198 | -0.220 |
| after T5 ped_wait |  |  |  |  | -0.107 | +0.030 | +0.017 |
| after T6 pickup |  |  |  |  |  | -0.010 | +0.191 |
| after T7 stationary |  |  |  |  |  |  | -0.108 |

## Aggregate All-Target ADE/FDE

| after | Normal ADE | PEGP ADE | ΔADE | Normal FDE | PEGP FDE | ΔFDE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T1 traffic_light_straight | 3.998 | 4.080 | +0.082 | 7.503 | 7.686 | +0.183 |
| T2 following | 4.707 | 4.736 | +0.029 | 9.189 | 9.320 | +0.131 |
| T3 high_lat | 4.962 | 5.191 | +0.229 | 9.269 | 9.809 | +0.540 |
| T4 near_multi | 5.285 | 5.461 | +0.176 | 9.792 | 10.370 | +0.578 |
| T5 ped_wait | 4.175 | 4.234 | +0.059 | 7.947 | 7.990 | +0.043 |
| T6 pickup | 7.236 | 7.005 | -0.231 | 14.328 | 13.678 | -0.650 |
| T7 stationary | 9.776 | 9.172 | -0.604 | 15.363 | 15.370 | +0.008 |

## Final Per-Task ADE/FDE

| task | Normal ADE | PEGP ADE | ΔADE | Normal FDE | PEGP FDE | ΔFDE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| traffic_light_straight | 13.379 | 11.736 | -1.643 | 21.649 | 20.315 | -1.334 |
| following | 11.919 | 11.379 | -0.540 | 18.298 | 19.113 | +0.815 |
| high_lat | 11.707 | 11.068 | -0.639 | 18.454 | 18.676 | +0.221 |
| near_multi | 14.626 | 13.937 | -0.689 | 22.293 | 22.638 | +0.346 |
| ped_wait | 7.063 | 7.076 | +0.013 | 10.953 | 11.619 | +0.666 |
| pickup | 7.707 | 7.085 | -0.623 | 12.173 | 11.549 | -0.624 |
| stationary | 2.033 | 1.925 | -0.108 | 3.720 | 3.683 | -0.037 |

## ADE Δ Matrix: PEGP - Normal

| after | traffic_light_straight | following | high_lat | near_multi | ped_wait | pickup | stationary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T1 traffic_light_straight | +0.196 | -0.019 | +0.232 | +0.074 | +0.138 | +0.014 | -0.061 |
| T2 following | +0.246 | +0.031 | +0.151 | +0.264 | -0.147 | -0.209 | -0.135 |
| T3 high_lat | +0.248 | +0.204 | +0.031 | +0.965 | +0.141 | -0.207 | +0.221 |
| T4 near_multi | +0.263 | +0.408 | -0.080 | +0.441 | +0.222 | +0.198 | -0.220 |
| T5 ped_wait | +0.090 | -0.365 | +0.444 | +0.304 | -0.107 | +0.030 | +0.017 |
| T6 pickup | -0.543 | -0.524 | -0.524 | +0.106 | -0.315 | -0.010 | +0.191 |
| T7 stationary | -1.643 | -0.540 | -0.639 | -0.689 | +0.013 | -0.623 | -0.108 |

## Matrix Files

- Normal: `/workspace2/z_project/exp/nuplan/nuplan_whitenoise_diffusion_v1/backfill_lora_cl7_upper_triangle_20260602/normal/upper_triangle_validation.md`
- PEGP: `/workspace2/z_project/exp/nuplan/nuplan_whitenoise_diffusion_v1/backfill_lora_cl7_upper_triangle_20260602/pegp/upper_triangle_validation.md`
- JSON: `/workspace2/z_project/exp/nuplan/nuplan_whitenoise_diffusion_v1/backfill_lora_cl7_upper_triangle_20260602/normal/upper_triangle_validation.json`, `/workspace2/z_project/exp/nuplan/nuplan_whitenoise_diffusion_v1/backfill_lora_cl7_upper_triangle_20260602/pegp/upper_triangle_validation.json`
