# Route B e20 Checkpoint Selection Reflection

Date: 2026-05-11

This note records the checkpoint-selection lesson from the
`route_b_next_token_modulation_graph_0511` run. The immediate context is that
e20 is being tested in closed loop and early routes look promising, even though
some offline metrics continue to improve after e20.

## Observation

The run shows a clear multi-task timing mismatch:

- Semantic state converges early, roughly around e15 to e20.
- Route and trajectory losses continue to improve later, but the useful route
  improvement becomes small after e20.
- Continuing to e25/e30 makes the state heads more conservative, especially
  window active and merge recall.
- Closed-loop behavior may prefer the earlier, less over-specialized state over
  the later lower-loss checkpoint.

Key metrics from the run:

| epoch | route L2 | route final L2 | window active recall | merge recall | none recall |
|---:|---:|---:|---:|---:|---:|
| 15 | 0.1559 | 0.2805 | 0.5767 | 0.6359 | 0.8850 |
| 20 | 0.1486 | 0.2646 | 0.5187 | 0.5688 | 0.9255 |
| 25 | 0.1551 | 0.2637 | 0.4660 | 0.4941 | 0.9395 |
| 30 | 0.1531 | 0.2635 | 0.4439 | 0.4688 | 0.9493 |

Route final L2 improves substantially from e15 to e20, but after e20 it is
almost flat. In contrast, merge recall continues to fall from e20 to e30.

## Interpretation

The important lesson is that the model does not always benefit from learning
every offline detail as tightly as possible.

For route, the gap between current e20 and the historical very low route-L2
runs looks large in relative terms, but small in absolute geometry:

- Historical independent/chase runs reached route L2 around `0.09`.
- Current e20 is `0.1486`.
- The difference is about `0.05` to `0.06` meters per route waypoint.
- Current e20 route final L2 is `0.2646`, while the best historical route final
  values are around `0.205`, again roughly a few centimeters apart.

For closed-loop control, a route waypoint error at this scale is likely less
important than whether the semantic state triggers the right behavior window,
especially merge, junction, borrow, go/yield, and chase/front-following. A
slightly less precise but smoother route may even be preferable if it avoids
overfitting to dataset-specific route details.

In other words, a checkpoint can be better for closed loop even if later
checkpoints have lower offline loss. The offline route metric should be treated
as a sanity check, not the only selection criterion.

## Current Checkpoint Preference

For this run, e20 is the most balanced candidate:

- e15 keeps window/merge more active, but route is weaker.
- e20 improves route enough while keeping state reasonably healthy.
- e25/e30 lower some losses, but state becomes too conservative and merge
  recall drops too much.

The current closed-loop test using e20 is therefore consistent with the metric
tradeoff: e20 may sit near the best practical point, even though it is not the
best route-L2 checkpoint.

## Training Implication

This supports a schedule-based multi-task strategy rather than simply training
longer:

- Train semantic state strongly in the early phase.
- After state reaches a useful regime, reduce or sparsify state updates.
- Let route/traj/speed continue learning without pushing state heads into an
  over-conservative solution.

A practical next config direction:

```yaml
train_stage1_until_epoch: 20  # or 25
train_stage1_after_update_every: 5  # or 10
route_loss_weight: 6.0  # maybe 8.0 for short fine-tune
stage1_window_weight: 0.10  # for route-focused resume only
stage1_phase_weight: 0.05
go_opportunity_loss_weight: 0.03
```

Longer-term, a cleaner solution is to freeze or lower the LR for semantic state
heads after e20-ish, while keeping route/traj/speed trainable.

## Route Coarse Memory Note

The closed-loop agent has been updated so route prediction now receives
`prev_route_coarse_memory` from the semantic cache:

```text
valid + prev window onehot(4) + prev dir onehot(4)
```

This reduces the earlier train/closed-loop mismatch where offline train/val had
GT previous coarse state but inference might not have passed any coarse memory.
There remains a softer mismatch: train/val still uses offline GT previous state,
while closed loop uses predicted cached state. Future training should consider
dropout/noise or scheduled sampling for `prev_route_coarse_memory`.

