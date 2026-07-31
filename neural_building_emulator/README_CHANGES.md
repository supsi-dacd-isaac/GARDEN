# Neural Building Emulator Changes Since Commit 6170836

This file summarizes the uncommitted emulator changes, ordered roughly by
expected likelihood of improving model performance on long open-loop building
temperature rollouts.

## Best Run So Far

Verbatim argument string reported as best-performing so far:

```text
--max-profiles 1000 --test-fraction 0.1 --epochs 20 --batch-size 32 --input-encoder-dim 8 --input-encoder-hidden-dim 64 --input-encoder-depth 2  --sequence-length 960  --state-dim 5 --schur-mode near_identity --schur-gamma 0.99 --target-mode absolute --num-full-profile-plots 5 --hidden-dim 32 --prob-particles 10 --prob-eval-particles 16 --prob-latent-dim 4 --prob-process-noise heteroscedastic --model-kind probabilistic --save-model --stride 2024  --zero-d --input-encoder-feedback thermal_gaps --switching-dynamicsheating_bias --input-feature-mode heating_regime --switching-alpha-heat-scale 5.0
```

Note: the string above is copied exactly. The current CLI spelling expects a
space between the flag and value:

```text
--switching-dynamics heating_bias
```

## Ranked Changes

### 1. Probabilistic stable state-space emulator

Files:

- `models/probabilistic_emulator.py`
- `train.py`
- `model_io.py`

Added `--model-kind probabilistic`, where each trajectory particle samples a
persistent latent vector `xi`, generates one stable system, and rolls out the
whole horizon with fixed sampled parameters. This is closer to scenario
propagation than a single deterministic mean model.

Why it is likely useful:

- Captures building/system uncertainty that a single deterministic model tends
  to average out.
- The reported best command uses this mode.
- Supports trajectory-level probabilistic losses instead of only pointwise MSE.

Main options:

- `--prob-particles`
- `--prob-eval-particles`
- `--prob-latent-dim`
- `--prob-process-noise {none,constant,heteroscedastic}`
- `--prob-softopt-weight`
- `--prob-variogram-weight`

### 2. Neural forcing encoder `z[t] = e(u[t])`

Files:

- `models/emulator.py`
- `models/probabilistic_emulator.py`
- `train.py`

Added an optional memoryless input encoder before the linear state-space
dynamics:

```text
z[t] = e(u[t])
x[t+1] = A x[t] + B z[t] + b
y[t] = C x[t] + D z[t] + d
```

Why it is likely useful:

- The thermal response to heating, solar, ventilation, and outdoor temperature
  is not purely linear in the raw input vector.
- The best-known run uses `--input-encoder-dim 8`.
- Keeps temporal memory in the state while allowing nonlinear instantaneous
  forcing.

### 3. Predicted-temperature feedback in the input encoder

Files:

- `models/emulator.py`
- `models/probabilistic_emulator.py`
- `train.py`
- `model_io.py`

Added:

```text
--input-encoder-feedback predicted_temperature
--input-encoder-feedback thermal_gaps
```

The stronger option, used in the best run, is:

```text
--input-encoder-feedback thermal_gaps
```

This augments the encoder input with:

```text
y_pred[t]
Tout[t] - y_pred[t]
Tset - y_pred[t]
```

Why it is likely useful:

- The same outdoor temperature has different meaning depending on current
  indoor temperature.
- It gives the encoder direct access to the thermal deficit without teacher
  forcing.
- It helps represent cooling/heating response rates as functions of current
  predicted state.

Constraints:

- Requires `--zero-d`.
- Currently requires `--target-mode absolute`.

### 4. Heating-regime input features

Files:

- `columns.py`
- `data.py`
- `train.py`

Added:

```text
--input-feature-mode heating_regime
```

This appends two features to the model inputs:

```text
Q_heat_is_on
Q_heat_recently_on_<window>
```

Why it is likely useful:

- The October failures look regime-dependent: free-floating behavior and
  controlled/heated behavior are different.
- `recently_on` gives the model memory of heating season/control activation
  without asking the state-space generator to infer it from a single heat value.
- The best-known run uses this option.

Related options:

- `--heating-regime-window-steps`
- `--heat-on-threshold`

### 5. Conservative switched dynamics: `heating_bias`

Files:

- `models/emulator.py`
- `models/probabilistic_emulator.py`
- `train.py`
- `model_io.py`

Added:

```text
--switching-dynamics heating_bias
```

This implements:

```text
A_t = A_shared
B_t = B_shared
b_t = (1 - alpha_t) b_off + alpha_t b_on
```

with:

```text
alpha_t = sigmoid((Q_heat - threshold) / heat_scale
                  + on_weight * (heating_on - 0.5)
                  + recent_weight * (recently_on - 0.5))
```

Why it is likely useful:

- Lets the model learn a different regime offset/forcing baseline when heating
  is active.
- Avoids the unstable products created by switching between different `A`
  matrices.
- The user-reported best run includes the intended `heating_bias` strategy.

Related options:

- `--switching-alpha-heat-scale`
- `--switching-alpha-on-weight`
- `--switching-alpha-recent-weight`

Important caveat:

- The older full switched mode `--switching-dynamics heating` is still present
  for comparison, but it can be unstable because it switches/interpolates
  `A`, `B`, and `b`.

### 6. Heat input normalization to W/m2

Files:

- `columns.py`
- `data.py`
- `train.py`

Added/standardized:

```text
--heat-input-normalization per_floor_area
```

Why it is likely useful:

- Heating power in W is not comparable across buildings with different floor
  areas.
- W/m2 is a more transferable input for a metadata-conditioned global model.

### 7. Longer rollout windows and wider seasonal coverage through stride

Files:

- `data.py`
- `train.py`

The windowing code supports long multi-step rollout windows and configurable
stride. The best-known run uses:

```text
--sequence-length 960
--stride 2024
```

Why it is likely useful:

- A 960-step window is about 10 days at 15-minute resolution.
- A large stride allows training on many more buildings while still sampling
  different parts of the year.
- This directly targets the long-horizon drift problem better than 96-step
  windows alone.

### 8. Heteroscedastic process noise

Files:

- `models/probabilistic_emulator.py`
- `train.py`

Added:

```text
--prob-process-noise heteroscedastic
```

Why it is likely useful:

- Allows uncertainty to grow under hard operating conditions, such as solar
  transients, abrupt heating changes, or poorly identified regimes.
- The best-known run uses this mode.

### 9. Checkpoint selection, early stopping, and model artifact saving

Files:

- `train.py`
- `model_io.py`

Added:

```text
--checkpoint-metric {auto,train_rmse_c,test_rmse_c}
--early-stopping-patience
--early-stopping-min-delta
--save-model
--save-model-every-epochs
--model-checkpoint-dir
```

Why it is likely useful:

- Several runs showed train loss improving while test/full-profile rollouts
  deteriorated badly.
- Selecting the best checkpoint by test RMSE avoids keeping the final overfit
  epoch.
- Saved artifacts enable ex-post analysis without retraining.

### 10. Full-profile evaluation and multiple interactive plots

Files:

- `train.py`

Added full-profile continuous rollout evaluation, in addition to fixed-window
test evaluation.

Added configurable plot counts:

```text
--num-window-plots
--num-full-profile-plots
```

For probabilistic models, full-profile plots show the sampled mean and a 95%
scenario band. Probabilistic plot filenames get the `_prob` suffix.

Why it is likely useful:

- The final emulator objective is simulator replacement over full years, not
  only 96-step windows.
- It exposes drift, winter/transition failures, and uncertainty calibration
  issues directly.

### 11. Loss normalization by per-window target standard deviation

Files:

- `train.py`

Added:

```text
--loss-normalization window_std
--loss-std-floor-c
```

Why it may help:

- Controlled winter periods often have tiny temperature variation but still
  contain important heating-response information.
- Normalizing by window standard deviation can prevent high-variance summer
  windows from dominating the objective.

Status:

- Implemented but not established as a best default yet.

### 12. Stable matrix parametrization options

Files:

- `models/schur.py`
- `train.py`

Added/extended:

```text
--schur-mode dense
--schur-mode near_identity
--schur-mode pf
--pf-lambda-min
```

Why it may help:

- `near_identity` gives a stronger prior for slow 15-minute thermal dynamics.
- `pf` gives a row-wise Perron-Frobenius/Gershgorin bound with nonnegative
  entries and row sums below `gamma`.

Status:

- The best-known run uses `--schur-mode near_identity --schur-gamma 0.99`.
- `pf` was useful to test but showed suspicious lower-bound behavior in some
  plots, so treat it as experimental.

### 13. Target reconstruction modes

Files:

- `train.py`

Added:

```text
--target-mode absolute
--target-mode residual
--target-mode delta
```

Why it may help:

- `residual` and `delta` were introduced to reduce sensitivity to poor initial
  state estimates.

Status:

- `delta` worsened predictions in the observed experiments.
- The best-known run uses `--target-mode absolute`.

### 14. Output timing and target alignment experiments

Files:

- `models/state_space.py`
- `models/emulator.py`
- `models/probabilistic_emulator.py`
- `data.py`
- `train.py`

Added:

```text
--output-timing {pre_update,post_update}
--target-alignment {same_time,next_step}
```

Why it may help:

- Tests whether EnergyPlus rows should be interpreted as beginning-of-interval
  or end-of-interval values.

Status:

- Single-building tests suggested `next_step` likely over-shifts the heat
  signal.
- Default remains the historical `same_time` + `pre_update`.

### 15. Physics-inspired monotonicity regularization

Files:

- `train.py`

Added:

```text
--monotonicity-weight
--monotonicity-horizon
--monotonicity-features
```

This penalizes negative sampled impulse responses for selected features such as
heat, outdoor temperature, and solar.

Status:

- Useful as a physics sanity constraint.
- Not currently implemented for probabilistic mode or feedback encoders.
- Did not by itself solve the heating-response issue.

## Diagnostic And Analysis Scripts

### Single-building direct state-space fitter

File:

- `fit_single_profile_ss.py`

Purpose:

- Fit one building directly, without metadata conditioning, to determine whether
  the state-space class itself can reproduce a profile.
- Supports full-year and period-sliced fitting through `--fit-start` and
  `--fit-end`.
- Useful for checking whether a failure is caused by the global neural
  generator or by the model class.

Main finding so far:

- Profile `745930` can fit Sep-Oct better when optimized only on that period,
  but still does not perfectly reproduce the sharp simulated behavior.

### Heat ablation analysis

File:

- `heat_ablation_analysis.py`

Purpose:

- Load a saved model artifact.
- Re-run selected full profiles while scaling or removing the heat input.
- Check whether the emulator actually reacts to heating power.

### Rollout signal diagnostics

File:

- `rollout_signal_diagnostics.py`

Purpose:

- Parse saved full-profile Plotly HTML rollouts.
- Compare simulated and emulated dynamics, including heat/temperature-change
  relationships.

## Known Caveats

1. The full switched mode `--switching-dynamics heating` can become unstable
   even when reported `rho_max` is below one, because stable individual matrices
   do not guarantee stable switched products.

2. Prefer `--switching-dynamics heating_bias` for now. It shares `A` and `B`
   and switches only `b`, so the state-transition matrix is fixed.

3. `rho_max` is still only a local diagnostic. For switched models it does not
   prove bounded full-year rollouts.

4. The exact best-args string above contains `--switching-dynamicsheating_bias`.
   The executable CLI flag is expected to be `--switching-dynamics heating_bias`.

5. Several options are diagnostic rather than recommended defaults:
   `--target-alignment next_step`, `--target-mode delta`, and `--schur-mode pf`
   should be treated cautiously.

## Verification Performed During Development

- `python -m compileall -q neural_building_emulator`
- deterministic smoke training with `--switching-dynamics heating`
- probabilistic smoke training with `--switching-dynamics heating`
- deterministic smoke training with `--switching-dynamics heating_bias`
- probabilistic smoke training with `--switching-dynamics heating_bias`
- saved probabilistic switched artifacts were loaded successfully through
  `model_io.load_training_artifact`
