# Code Review: S → M1 State-Motion Adapter

Date: 2026-05-21
Plan ref: `docs/tmp/state_motion_s_to_m1_adapter_plan_20260521.md`

## Overall Assessment

The staged skeleton (S/M1/A/M2) is fully implemented with config switches. The zero-init
strategy for the adapter preserves exact initial equivalence. Core freeze and partial-init
logic is sound. The following issues are ranked by severity.

---

## B1 — Freeze Guard Uses OR Instead of Per-Module Logic

File: `policy/annealed_energy_guidance_policy.py` ~L824

```python
if self.freeze_motion_backbone_for_adapter or self.freeze_semantic_state_for_adapter:
    for param in self.model.parameters():
        param.requires_grad_(False)
for name, param in self.model.named_parameters():
    if 'state_motion_adapter' in name:
        param.requires_grad_(True)
```

`or` means: if **either** flag is True, **all** model params are frozen first, then only
adapter params are unfrozen. A future config like `freeze_motion_backbone=True,
freeze_semantic_state=False` would still freeze S params — the two flags lose independent
meaning.

Current M1 config has both True so there is no immediate impact, but the two-flag design
implies per-module granularity.

**Fix (before M2):** Use separate freeze passes or document that the flags are only
meaningful together.

---

## B2 — Adapter Modules Always Instantiated

File: `model/transformer_for_diffusion_multi_head.py` ~L2248–2272

`state_motion_adapter_condition_proj`, both heads, and `global_gate` are created
unconditionally at the end of `__init__`, regardless of `use_state_motion_adapter`. Every
checkpoint — including S-stage and nostate G — will contain adapter keys with zero-init
values.

Consequences:
- ~3 × n_emb² extra parameters allocated always.
- If nostate G was built from this same codebase, its adapter keys (zero-init) will be
  partially loaded into S, which is harmless but potentially confusing.

**Fix (low urgency):** Add `if self.use_state_motion_adapter:` guard around lines 2248–2271
if memory matters.

---

## D1 — Alignment Critic: 3 of 5 Dimensions Are Identical `interval_score`

File: `policy/annealed_energy_guidance_policy.py` ~L3467

```python
return torch.stack([
    interval_score,        # total
    interval_score,        # phase-speed  ← same as total
    torch.minimum(lower_score, upper_score),
    upper_score,
    interval_score,        # interval-core ← same as total
], dim=-1)
```

Plan specified 5 distinct outputs (total, phase_speed, edge_speed, chase_speed,
route_window). Current impl collapses phase_speed and interval_core into total; route_window
is not implemented. With 3 identical dimensions the critic has effectively 3 independent
outputs.

**Fix (before A training):** Implement distinct targets for each dimension, or reduce
`alignment_score_dim` to 3 to avoid the false appearance of 5 independent signals.

---

## D2 — Risky-Passable Samples Are Both Soft-Targeted AND Downweighted

File: `policy/annealed_energy_guidance_policy.py` ~L3563

The plan says risky samples get a medium target (floor 0.55). Impl correctly sets
`risky_floor=True` in `pos_target`. But the same samples are also downweighted to 0.25:

```python
sample_weight = torch.where(
    risky,
    sample_weight * float(self.alignment_risky_passable_weight),  # 0.25
    sample_weight,
)
```

Combined effect: soft target + ¼ loss weight → A barely updates on ambiguous cases, which
may prevent it from learning the risky-passable distinction at all.

**Fix (before A training):** Either remove the weight scaling, or set
`alignment_risky_passable_weight > 1.0` to emphasize hard cases.

---

## D3 — Gate Sparsity Loss Pulls a 1.0-Init Gate Toward Zero from Step 1

File: `policy/annealed_energy_guidance_policy.py` ~L5419

```python
adapter_gate_sparsity_loss = gate.float().abs().mean()
```

Gate initializes to `1.0`. `gate_sparsity_weight=0.001` imposes a constant `0.001` penalty
pulling the gate toward zero from the very first step, competing with `motion_loss` gradients
that need a non-zero gate.

The zero-final-projection / gate=1.0 init is correct (plan implementation note confirmed).
The risk is early gate collapse before meaningful adapter gradients develop.

**Action:** Monitor `state_motion_adapter_gate` in W&B during M1 warmup (first ~500 steps).
If gate drops below 0.5 before losses stabilize, increase `state_motion_adapter_loss_weight`
or reduce `state_motion_adapter_gate_sparsity_weight`.

---

## M1 — M2 Alignment Energy Does Not Backprop Into the Adapter

File: `policy/annealed_energy_guidance_policy.py` ~L3540

```python
pos_features = self._build_alignment_critic_features(...).detach()
neg_features = self._build_alignment_critic_features(...).detach()
```

Both feature vectors are `.detach()`-ed before entering the critic. For A-stage this is
correct (critic-only updates). But in M2 (`adapter_m2` + `use_alignment_critic_for_adapter=
True`), `alignment_weighted_loss` is added to `state_motion_adapter_total_loss` yet **no
gradient flows back into the adapter or condition_proj** — the alignment loss only trains
critic weights even in M2.

**Fix (before M2):** If the intent is to push the adapter via alignment energy, remove the
detach on `pred_features` (the block ~L3590 computing pred_scores) and route it through a
differentiable path from the adapter's speed output.

---

## M2 — Script: `SKIP_S=1 SKIP_M1=1` Exits Silently with Code 0

File: `scripts/codex_bash/train_state_motion_s_to_m1.sh`

With both skip flags set, nothing runs, exit code 0, no warning. Easy to misuse.

**Fix (minor):** Add a check at the top:
```bash
if [[ "${SKIP_S}" == "1" && "${SKIP_M1}" == "1" ]]; then
  echo "Both SKIP_S=1 and SKIP_M1=1 set; nothing to do." >&2
  exit 1
fi
```

---

## Config Checklist

| Config                        | mode               | use_adapter | use_critic | freeze_backbone | align_weight |
|-------------------------------|--------------------|-------------|------------|-----------------|--------------|
| semantic_detached_0521        | semantic_detached  | false ✓     | false ✓    | true (unused)   | 0.0 ✓        |
| adapter_m1_0521               | adapter_m1         | true ✓      | false ✓    | true ✓          | 0.0 ✓        |
| alignment_critic_0521         | alignment_critic   | —           | true ✓     | —               | 1.0 ✓        |
| adapter_m2_0521               | adapter_m2         | true ✓      | true ✓     | true ✓          | 0.02 ✓       |

All configs match the plan's intent. Dataset path and checkpoint dirs consistently set. ✓

---

## Initialization Flow (Confirmed Correct)

Order in `TransformerForDiffusion.__init__`:

1. All modules defined (including adapter unconditionally — see B2)
2. `self.apply(_init_weights)` — single call, randomizes all weights
3. `_zero_init_state_motion_adapter()` — zeroes last `nn.Linear` of each head

With `gate=1.0` and `final_linear=zeros`, initial adapter delta is exactly zero. ✓
Partial init from nostate/S checkpoint handles key-mismatch gracefully via shape-matched
loading in `train_carla_bev.py` ~L2318. ✓

---

## Priority for Tonight's Run

| Issue | Action before S→M1 |
|-------|-------------------|
| D3 gate monitoring | Watch `state_motion_adapter_gate` in W&B during M1 warmup |
| B1 freeze OR semantics | Low risk; both flags are True in M1 config — no action needed tonight |
| B2 unconditional adapter | No correctness impact for nostate→S→M1 flow — no action needed tonight |
| M1 silent skip | Minor; no action needed tonight |
| D1, D2, M1 (M2 backprop) | A/M2-stage concerns only — defer |
