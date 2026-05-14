# Semantic Motion Condition Ablation - 2026-05-14

## Goal

Add a parameter switch for the next-token branch to ablate which semantic state
groups are injected into the traj/speed motion branch.

This only changes semantic-state-to-motion conditioning. It does not remove any
semantic heads, losses, labels, metrics, route-specific conditioning, or the
explicit route intent token.

## Config

```yaml
route_b:
  semantic_motion_condition_profile: all
```

Supported profiles:

```text
all
  B0 baseline. Keep current behavior.

window_only
  B0.25. Inject only window.

window_decision
  B0.5. Inject only window + decision_phase.

window_decision_control
  B1. Inject window + decision_phase + control_phase.

window_phase_opportunity
  B2. Inject window + decision/control phase + go_opportunity/yld_pressure.

compact_safe
  B3. Inject window + decision/control phase + opportunity + cover-edge speed constraints.
```

For `semantic_motion_condition_mode=compact_graph`, `compact_safe` keeps:

```text
window
decision_phase
control_phase
go_opportunity / yld_pressure
current/future edge valid + mode
edge speed margins
edge speed valid flags
```

It masks out:

```text
borrow_time
dir
area_status / timing
temporary occupancy bins
raw family boundary
```

## Implementation Note

The branch condition tensor layout and dimension stay unchanged. Disabled
groups are masked after their projector, at the embedding contribution level:

```text
condition_embedding += projector(group) * schedule_gate * profile_mask
```

This avoids a subtle issue where setting probabilities to zero would still allow
the projector bias to affect motion.

## Experiment Ladder

```text
B0    all
B0.25 window_only
B0.5  window_decision
B1    window_decision_control
B2    window_phase_opportunity
B3    compact_safe
```

Key comparisons:

```text
B0.25 -> B0.5: decision_phase usefulness
B0.5  -> B1:   control_phase usefulness
B1    -> B2:   opportunity prior usefulness
B2    -> B3:   edge/chase speed constraint usefulness
B0    -> B3:   full state vs compact causal state
```

## Metrics To Watch

```text
route_final / route_L2
speed MAE
chase metrics
window active recall
dir cross recall
decision/control go-yld recall
close-loop collision and route completion
```
