# Semantic-State GRPO-Like RL + Confidence Advantage

Date: 2026-05-05

## Motivation

We want to explore RL over semantic state instead of raw trajectory / speed.

This is analogous to LLM RL with GRPO:

- LLM action space: real text tokens.
- Our action space: structured semantic-state pseudo-text.

Our setting should be easier than LLM text RL because the state is small,
typed, and physically interpretable:

```text
window
dir
conflict area
area status / timing
temporary occupancy
go opportunity / yld pressure
decision_phase / control_phase
boundary speeds
chase state
```

The key idea is to use semantic state `z_t` as the RL action-like object.

## GRPO-Like Formulation

For the same observation `obs_t`, construct or sample a group of candidate
semantic states:

```text
z_1, z_2, ..., z_K
```

Candidates can come from:

- GT state
- current model prediction
- rule-corrected state
- counterfactual yld/go state
- perturbed boundary / tempocc / chase state
- next-token predicted state

Each candidate receives a short-horizon semantic reward:

```text
R_i = reward(obs_t, z_i)
```

Then compute group-relative advantage:

```text
A_i = normalize_group(R_i)
```

A simple objective:

```text
loss = - sum_i A_i * log p(z_i | obs_t)
       + beta * KL(p || p_ref)
```

This follows the spirit of GRPO:

- no value network required in V1
- normalize reward within the candidate group
- update the semantic policy toward relatively better states
- keep a KL / supervised-reference term to prevent drift

## Why Semantic State Is Easier Than Text

LLM RL is hard because text is open-ended and reward is sparse / fuzzy.

Here the pseudo-text state is structured:

- categorical heads have fixed classes
- route mask / temp-occ bins are small binary vectors
- boundary / timing / chase values are low-dimensional scalars
- many semantic contradictions can be scored by rules

Example rule scores:

```text
no area -> no tempocc occupancy
inside area -> phase/control should prefer go
occupied around arrival -> go opportunity should be low
go_min near cap -> phase go confidence should be low
low chase cap -> suppress aggressive go/speed
phase=go but chase cap too low -> penalty
phase=yld while clear gap persists -> mild penalty
```

This gives dense, interpretable reward before online CARLA RL.

## Confidence-Aware Advantage

A useful extension is to make advantage depend not only on reward, but also on
semantic confidence.

Naive version:

```text
A_i = normalize_group(R_i) + lambda * normalize_group(C_i)
```

But `C_i` should not be raw model softmax confidence only. Otherwise the model
may learn to be confidently wrong.

Preferred confidence is evidence-supported confidence:

```text
C_i =
  noise-view / ensemble agreement
  + rule consistency margin
  + temporary occupancy evidence strength
  + boundary feasibility margin
  + chase safety margin
  - contradiction penalty
```

Examples:

- `phase=go` with reachable `go_min`, clear temp-occ, and safe chase cap gets a confidence bonus.
- `phase=go` with `go_min ~= speed_cap` gets a confidence penalty.
- `phase=yld/go` stable across multiple noise views gets a confidence bonus.
- Ambiguous dual-feasible scenes should not be forced into low entropy too early.

So confidence is not simply "low entropy is good". The desired behavior is:

```text
clear evidence -> confident state
ambiguous evidence -> calibrated uncertainty
contradictory evidence -> lower confidence
```

## Relationship To Current Problems

This directly targets several observed failure modes:

- `decision_phase=go` while `go_min_speed` is effectively unreachable.
- `go_opportunity_prob` too conservative or too narrow.
- yld/go phase jitter across nearby frames.
- temp-occ bins flicker even when the physical situation is similar.
- chase state predicts a high cap when lead vehicle is actually slow.

The semantic reward can make these contradictions explicit without directly
optimizing raw trajectory.

## Offline V1 Recommendation

Before online RL, use offline GRPO-like reward distillation:

1. For each sample, build a small candidate set around the current semantic label.
2. Score candidates with rule-based short-horizon semantic reward.
3. Compute group-relative advantage.
4. Train the semantic-state policy with weighted log-prob / ranking loss.
5. Keep supervised GT CE/SmoothL1 as anchor.

Suggested V1 loss:

```text
loss =
  supervised_state_loss
  + w_grpo * group_relative_state_loss
  + w_kl * KL(policy || supervised_ref)
  + w_conf * confidence_calibration_loss
```

The first implementation should stay conservative:

- no Q-learning
- no long-horizon value target
- no raw traj/speed RL
- no online simulator dependency

## Later Online RL Interface

For CARLA online RL, keep the policy action space as semantic state:

```text
obs_t -> choose / adjust z_t
z_t -> supervised traj / route / speed planner
sim -> short-horizon reward
```

This should be more sample-efficient than optimizing raw trajectory because the
state space is smaller and already decomposed into meaningful factors.

## Open Questions

- Should confidence be added into advantage, or trained as a separate calibrated head?
- Should group candidates be generated by rules, noise sampling, or both?
- Should continuous values like boundary / chase speed be discretized for log-prob training?
- How strong should KL-to-supervised-reference be?
- How do we avoid penalizing valid dual-mode scenes where both yld and go are acceptable?
