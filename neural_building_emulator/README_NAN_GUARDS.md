# NaN Guard And Diagnostic Changes

This note tracks the changes introduced after the closed-loop HP probabilistic
training started producing NaNs. Its purpose is twofold:

1. Make the current numerical-safety behavior explicit before committing.
2. Define a clear rollback boundary if these changes make results worse.

The changes below are focused on preventing or diagnosing NaNs. They do not
claim to solve the modeling issue by themselves.

## Files Touched By The NaN Guard Bundle

- `neural_building_emulator/train.py`
- `neural_building_emulator/models/probabilistic_closed_loop_hp.py`
- `neural_building_emulator/models/contracting_closed_loop_hp.py`
- `neural_building_emulator/models/probabilistic_contracting_closed_loop_hp.py`
- `neural_building_emulator/model_io.py`
- `neural_building_emulator/flexibility_event_study_kpis.py`

Some of these files also contain unrelated model-development changes from the
same working period. The sections below identify the NaN-specific parts.

## New Optimizer Guard Path

The training script now builds optimizers through `build_optimizer(config)`
instead of directly constructing Adam in the relevant training branches.

Implemented in:

- `train.py::build_optimizer`

Behavior:

- Optional global gradient clipping is applied before Adam.
- Adam is still the base optimizer.
- By default, the optimizer is wrapped with `optax.apply_if_finite`.

CLI flags:

- `--gradient-clip-norm FLOAT`
  - Default: `1.0`.
  - Applies `optax.clip_by_global_norm`.
  - Use `0` to disable clipping.
- `--allow-nonfinite-updates`
  - Default behavior is to skip non-finite updates.
  - Passing this flag disables that guard.
- `--max-consecutive-nonfinite-updates INT`
  - Default: `8`.
  - Passed to `optax.apply_if_finite`.
  - If too many consecutive non-finite updates occur, Optax can raise instead
    of silently continuing.

Important limitation:

- `optax.apply_if_finite` guards optimizer updates, but it does not by itself
  explain where a NaN originated.
- That is why the explicit finite-flag diagnostics below were added.

## Explicit Finite-Update Checks

The closed-loop training steps now compute a candidate update first and inspect
whether the relevant quantities are finite before accepting it.

Implemented in:

- `train.py::_array_tree_all_finite`
- `train.py::_select_array_tree`
- `train.py::_update_finite_flags`
- `train.py::closed_loop_train_step`
- `train.py::probabilistic_closed_loop_train_step`

The finite flags are reported in this order:

```text
loss
components
grads
updates
opt_state
params
forward
```

Meaning of each flag:

- `loss`: the scalar objective is finite.
- `components`: probabilistic loss components are finite.
- `grads`: all gradient leaves are finite.
- `updates`: all optimizer update leaves are finite.
- `opt_state`: the candidate optimizer state is finite.
- `params`: the candidate model parameters are finite.
- `forward`: optional candidate rollout is finite.

When `skip_nonfinite_updates=True`:

- If all flags are true, the candidate model and optimizer state are accepted.
- If any flag is false, the previous model and optimizer state are retained.

When `--allow-nonfinite-updates` is passed:

- The candidate update is accepted even if the explicit finite flags fail.
- This is mainly useful for reproducing old NaN-prone behavior.

## Candidate Forward Validation

A separate optional check was added for the case where parameters remain finite
but the updated model would already produce a non-finite rollout.

CLI flag:

- `--validate-candidate-updates`

Implemented in:

- `train.py::closed_loop_train_step`
- `train.py::probabilistic_closed_loop_train_step`

Behavior:

- After computing the candidate updated model, the code runs one extra forward
  rollout on the current minibatch.
- If that candidate rollout is non-finite, the `forward` flag fails.
- If skipping is enabled, that update is rejected.

Probabilistic detail:

- For probabilistic closed-loop models, candidate validation calls
  `predict_probabilistic_closed_loop_batch_with_aux`.
- It uses the same minibatch and particle count as the training step.
- It uses `hp_scenario_mode="expected"` for the candidate check.
- It samples thermal process noise according to the model configuration.

Cost:

- This can be noticeably slower because it adds an extra rollout per minibatch.
- It is meant for diagnostic or high-value long runs, not necessarily every
  production training run.

## Update Diagnostics In Epoch Logs

The closed-loop training loop now tracks update diagnostics per minibatch and
prints per-epoch maxima when requested, or whenever an update is skipped.

CLI flag:

- `--log-update-diagnostics`

Implemented in:

- `train.py::_array_tree_global_norm`
- `train.py::_array_tree_max_abs`
- `train.py::_update_stats`
- closed-loop epoch logging in `train.py::run_closed_loop_training`

Logged quantities:

```text
max_grad_norm
max_update_norm
max_param_norm
max_max_abs_update
max_max_abs_param
```

Interpretation:

- `max_grad_norm`
  - Maximum global gradient norm observed during the epoch.
  - If this spikes before NaNs, the backward pass is likely the first issue.
- `max_update_norm`
  - Maximum global optimizer update norm observed during the epoch.
  - If gradients are modest but this spikes, Adam state/update dynamics are
    suspicious.
- `max_param_norm`
  - Maximum global parameter norm of candidate models.
  - Useful for spotting slow parameter drift.
- `max_max_abs_update`
  - Largest absolute single-parameter update in the epoch.
- `max_max_abs_param`
  - Largest absolute single parameter value in candidate models.

Skipped-update diagnostics:

```text
skipped_update_batches=N
first_skipped_update_batch=M
skipped_update_reasons=[grads:...,forward:...]
```

Interpretation:

- `grads`
  - NaNs/Infs are already present in gradients.
  - Likely causes include singular derivatives, unstable exponentials, or a
    loss term with bad local gradient behavior.
- `updates`
  - Gradients may be finite, but the optimizer update is not.
  - Usually points to optimizer-state issues.
- `params`
  - Candidate parameters contain NaN/Inf.
- `forward`
  - Candidate parameters are finite, but the updated model produces a
    non-finite rollout on the same minibatch.
  - This suggests a finite but dynamically dangerous update.

Loss logging changes:

- Epoch train losses now use a finite-aware mean through `finite_mean`.
- If some minibatch losses are non-finite, the epoch line adds:

```text
nonfinite_train_batches=N
```

- Probabilistic closed-loop component means use `finite_column_means`.
- If any component rows are non-finite, the epoch line adds:

```text
nonfinite_component_batches=N
```

## Smooth Contracting Matrix Normalization

The contracting closed-loop models normalize a generated matrix so its
Frobenius norm is bounded before applying the recurrent transition.

The previous form used a direct matrix norm:

```python
frobenius = jnp.linalg.norm(raw_matrix)
divisor = jnp.maximum(frobenius, 1.0)
matrix = contraction_gamma * raw_matrix / divisor
```

This is finite in the forward pass, but the derivative of a norm at exactly or
near zero can be numerically problematic. A model can therefore have a finite
loss and still produce NaN gradients.

The new form is:

```python
frobenius = jnp.sqrt(jnp.sum(raw_matrix**2) + 1e-12)
divisor = jnp.maximum(frobenius, 1.0)
matrix = contraction_gamma * raw_matrix / divisor
```

Implemented in:

- `models/contracting_closed_loop_hp.py::transition_matrix_and_bias`
- `models/probabilistic_contracting_closed_loop_hp.py::transition_matrix_and_bias`

What this preserves:

- The same practical contraction bound:

```text
||matrix||_2 <= ||matrix||_F <= contraction_gamma
```

What this changes:

- The norm is smooth at zero because of the epsilon inside the square root.
- This removes one plausible source of NaN gradients without changing the
  intended bounded-matrix parametrization.

## Bounded HP Electric-Power Emission

The probabilistic closed-loop HP models now support an explicit HP emission
mode:

```text
ProbHpEmissionMode = {"bounded", "legacy_lognormal_mean"}
```

Implemented in:

- `models/probabilistic_closed_loop_hp.py`
- `models/probabilistic_contracting_closed_loop_hp.py`
- `train.py`
- `model_io.py`
- `flexibility_event_study_kpis.py`

CLI flag:

- `--prob-hp-emission-mode {bounded,legacy_lognormal_mean}`
  - Default for new training: `bounded`.

### Bounded Mode

In bounded mode, active HP electric power is represented in log space, but both
the log mean and log standard deviation are explicitly bounded.

The active-power log mean is:

```python
max_log_active = log1p(Pmax)
log_mu = max_log_active * sigmoid(raw_mu)
```

So:

```text
0 <= log_mu <= log1p(Pmax)
```

The log standard deviation is:

```python
log_sigma = 0.05 + 0.45 * sigmoid(raw_sigma)
```

So:

```text
0.05 <= log_sigma <= 0.50
```

The expected active power used in the differentiable path is:

```python
expected_active = expm1(clip(log_mu, 0, log1p(Pmax)))
expected_total = pi * expected_active
```

Important modeling/numerical consequence:

- The expected active power no longer uses
  `expm1(log_mu + 0.5 * log_sigma**2)`.
- This intentionally removes the variance-driven exponential blow-up from the
  differentiable thermal/buffer path.

Scenario sampling:

- A sampled active log power is still formed as:

```python
sampled_log_active = log_mu + log_sigma * noise
```

- But sampled log power is clipped to `[0, log1p(Pmax)]` before converting back
  to W/m2.
- Therefore sampled active power is also bounded.

The power cap `Pmax` is:

- `hp_pel_cap_w_m2` when explicitly provided, otherwise
- a train-data-derived fallback based on target scaling:

```text
max(mean + 8 * scale, 2 * scale, 1e-3)
```

### Legacy Mode

The old behavior remains available as:

```bash
--prob-hp-emission-mode legacy_lognormal_mean
```

Legacy behavior:

- `log_sigma = 0.05 + 0.70 * sigmoid(raw_sigma)`.
- Expected active power uses:

```python
expm1(log_mu + 0.5 * log_sigma**2)
```

- The expected value is capped afterwards.

Reason to keep it:

- It allows reproduction of older runs.

Reason it is risky:

- The exponential mean path can create very large gradients even when the final
  value is later capped.

## Artifact Loading And Backward Compatibility

Saved artifacts now carry/use `prob_hp_emission_mode` through training config.

Implemented in:

- `model_io.py`

Backward compatibility:

- Old artifacts that do not contain `prob_hp_emission_mode` default to
  `legacy_lognormal_mean` when loaded.
- This avoids silently changing the behavior of old checkpoints.

Flexibility KPI script support:

- `flexibility_event_study_kpis.py` accepts:

```bash
--prob-hp-emission-mode artifact
--prob-hp-emission-mode bounded
--prob-hp-emission-mode legacy_lognormal_mean
```

Behavior:

- `artifact` uses the mode stored in the saved artifact.
- `bounded` or `legacy_lognormal_mean` overrides the loaded training config
  before model construction.

The script also records the resolved HP emission mode in:

- `flexibility_event_study_summary.json`
- HTML subtitles
- terminal output

## Recommended Diagnostic Command Additions

For a long run where NaNs are the main concern, add:

```bash
--prob-hp-emission-mode bounded \
--gradient-clip-norm 1.0 \
--log-update-diagnostics \
--save-model-every-epochs 1
```

If the extra runtime is acceptable, also add:

```bash
--validate-candidate-updates
```

Tradeoff:

- `--validate-candidate-updates` is the most informative but slowest addition.
- `--log-update-diagnostics` is cheap and should be useful in most long runs.
- `--save-model-every-epochs 1` is not a NaN guard, but it is useful insurance
  for multi-hour training runs because it preserves intermediate artifacts.

## How To Interpret The Next Failure

If the next run still produces NaNs:

### Case 1: `skipped_update_reasons=[grads:...]`

The forward pass produced a finite loss, but differentiation produced NaN/Inf.
Likely suspects:

- nonsmooth operations;
- exponentials;
- loss terms with bad local gradients;
- remaining norm-like operations at zero.

The smooth Frobenius patch was specifically aimed at this case.

### Case 2: `skipped_update_reasons=[updates:...]`

Gradients were finite but optimizer updates were not. Likely suspects:

- Adam accumulator state;
- learning rate too high;
- interaction between clipping and optimizer state.

### Case 3: `skipped_update_reasons=[params:...]`

The candidate model parameters became NaN/Inf after applying the update.
This usually means the update itself is too large or already non-finite.

### Case 4: `skipped_update_reasons=[forward:...]`

The candidate model parameters are finite, but the candidate rollout is not.
This is a dynamically dangerous finite update. In that case, the most relevant
fields are:

```text
max_update_norm
max_max_abs_update
first_skipped_update_batch
```

### Case 5: no skipped updates, but evaluation becomes NaN

If candidate validation is disabled, a finite-parameter update may still be
accepted and only fail on a later train/eval rollout. Re-run with:

```bash
--validate-candidate-updates
```

If candidate validation is enabled and this still happens, the candidate
minibatch was finite but another window/profile was not. Then the next useful
diagnostic is to identify the profile/window of first non-finite evaluation.

## Verification Performed

After these changes, the following checks were run:

```bash
.venv/bin/python -m py_compile \
  neural_building_emulator/train.py \
  neural_building_emulator/models/contracting_closed_loop_hp.py \
  neural_building_emulator/models/probabilistic_contracting_closed_loop_hp.py
```

A one-batch smoke test was also run for
`closed_loop_hp_contracting_probabilistic` with:

```bash
--max-profiles 8
--epochs 1
--max-train-batches 1
--prob-particles 2
--prob-eval-particles 2
--prob-process-noise none
--prob-hp-emission-mode bounded
--gradient-clip-norm 1.0
--validate-candidate-updates
--log-update-diagnostics
```

The smoke test completed and emitted finite update diagnostics.

## Rollback Checklist

If we decide to revert only the NaN guard bundle, the likely rollback targets
are:

1. Remove optimizer guard CLI/config fields from `TrainConfig` and `parse_args`:
   - `gradient_clip_norm`
   - `skip_nonfinite_updates`
   - `max_consecutive_nonfinite_updates`
   - `validate_candidate_updates`
   - `log_update_diagnostics`

2. Remove or bypass `build_optimizer` and return to the previous direct Adam
   construction in training branches.

3. Remove helper functions from `train.py`:
   - `_array_tree_all_finite`
   - `_array_tree_global_norm`
   - `_array_tree_max_abs`
   - `_select_array_tree`
   - `_update_stats`
   - `_update_finite_flags`

4. Revert closed-loop train-step return signatures and call sites:
   - `closed_loop_train_step`
   - `probabilistic_closed_loop_train_step`
   - closed-loop epoch logging around skipped updates and update stats.

5. Revert smooth Frobenius normalization if desired:
   - `sqrt(sum(raw_matrix**2) + 1e-12)`
   - back to the prior norm expression.

6. Revert bounded HP emission if desired:
   - remove `ProbHpEmissionMode`;
   - remove `--prob-hp-emission-mode`;
   - restore the previous lognormal-mean behavior as the only path.

7. Remove artifact/script compatibility for `prob_hp_emission_mode`:
   - `model_io.py`;
   - `flexibility_event_study_kpis.py`.

This file itself can then be deleted.
