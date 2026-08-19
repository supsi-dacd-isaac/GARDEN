# Neural Building Emulator Models

This document summarizes the emulator model families exposed by:

```bash
.venv/bin/python -m neural_building_emulator.train
```

There are two main modeling strategies:

1. **Q-to-T thermal emulator**: delivered room heat is known and provided as an
   exogenous input.
2. **Closed-loop HP emulator**: delivered room heat and HP electric power are
   predicted from setpoint/weather/current temperature through a learned
   controller, buffer, and HP model.

The Q-to-T emulator is the simpler base case and is described first. The
closed-loop HP model is then described incrementally: only the extra assumptions,
targets, states, and losses are repeated.

## Model Kind Overview

Q-to-T models:

```bash
--model-kind deterministic
--model-kind probabilistic
```

Closed-loop HP models:

```bash
--model-kind closed_loop_hp
--model-kind closed_loop_hp_contracting
--model-kind closed_loop_hp_probabilistic
--model-kind closed_loop_hp_contracting_probabilistic
```

Current rough status:

```text
deterministic                  stable Q-to-T baseline
probabilistic                  probabilistic Q-to-T baseline
closed_loop_hp                 current strongest closed-loop baseline
closed_loop_hp_contracting     data-driven contractive closed-loop state model
closed_loop_hp_probabilistic   active but uncertainty calibration is still experimental
closed_loop_hp_contracting_probabilistic
                               probabilistic contractive model with leaky buffer
```

## Shared Dataset Context

The current default dataset path is:

```text
neural_building_emulator/tessin_results.parquet/variable_setpoints
```

The modeled apartment/zone is the first zone:

```text
FL0_THZ0
```

The main indoor temperature target column is:

```text
FL0_THZ0, Zone Air Temperature
```

The common disturbance columns are:

```text
Environment, Site Outdoor Air Drybulb Temperature
Environment, Site Global Horizontal Solar Radiation Rate per Area
FL0_THZ0, Zone Ventilation Standard Density Volume Flow Rate
```

Static metadata used by the state-space generators:

```text
buildingType
totalFloors
constructionPeriod
wallsRenovationPeriod
floorsRenovationPeriod
roofRenovationPeriod
windowsRenovationPeriod
dwellingNumber
shSetpoint
floor_area
volume
envelope_area
window_area
surface_to_volume_ratio
thermal_mass_class
```

Metadata, inputs, and targets are standardized from training windows. Evaluation
and plots are inverse-transformed back to physical units.

# Q-to-T Thermal Emulator

The Q-to-T model predicts indoor temperature from known room heat and known
disturbances:

```text
known Qroom[t], weather[t], solar[t], ventilation[t], metadata
  -> predicted Tin[t]
```

This is the right model when the heat entering the modeled apartment is already
known or provided by an external controller/simulator.

The recommended heat input is:

```bash
--heating-mode zone_thermal
```

or equivalently:

```bash
--heating-mode A
```

which uses:

```text
zone_thermal_heating_power
```

The model can also be run with:

```bash
--heating-mode heating_electric
```

or:

```bash
--heating-mode B
```

but then the first input is HP electric power, not heat entering the zone. That
is a different and less clean identification problem because the HP, hydronic
buffer, and controller dynamics are hidden.

## Q-to-T Inputs

For the recommended setup, the raw input vector is:

```text
u[t] = [
  zone_thermal_heating_power[t],
  Tout[t],
  solar[t],
  ventilation[t],
]
```

By default the heat channel is normalized by floor area:

```bash
--heat-input-normalization per_floor_area
```

so the first input becomes:

```text
zone_thermal_heating_power_per_m2
```

With:

```bash
--input-feature-mode heating_regime
```

two regime features are appended:

```text
heat_is_on
heat_recently_on
```

where `heat_recently_on` is a rolling max over:

```bash
--heating-regime-window-steps
```

These features are mainly used by the experimental switching dynamics.

## Q-to-T Alignment

The standard target alignment is:

```bash
--target-alignment same_time
```

which trains:

```text
u[t] -> Tin[t]
```

The alternative is:

```bash
--target-alignment next_step
```

which trains:

```text
u[t] -> Tin[t+1]
```

The initial true temperature is supplied at the start of each rollout window:

```text
Tin[start]
```

## Deterministic Q-to-T State-Space Model

With:

```bash
--model-kind deterministic
```

metadata generates one stable state-space system per building:

```text
theta = theta_scale * theta_net(c)
```

The generated parameters define:

```text
A, B, C, D, b, d
```

The latent state update is:

```text
x[t+1] = A x[t] + B z[t] + b
```

The output can be pre-update or post-update:

```text
pre_update:  y[t] = C x[t]   + D z[t] + d
post_update: y[t] = C x[t+1] + D z[t] + d
```

The default is:

```bash
--output-timing pre_update
```

The initial state is generated from metadata and the true initial temperature:

```text
x[0] = x0_net(c, Tin_initial)
```

The matrix `A` is stable by construction using one of:

```bash
--schur-mode dense
--schur-mode near_identity
--schur-mode pf
```

and the spectral-radius upper bound:

```bash
--schur-gamma
```

`near_identity` is often a useful prior for 15-minute building thermal dynamics
because it initializes slow memory.

## Q-to-T Input Encoder

Without an input encoder:

```text
z[t] = u[t]
```

With:

```bash
--input-encoder-dim N
```

the model uses a memoryless neural forcing encoder:

```text
z[t] = encoder(u[t])
```

The encoder has no temporal memory. Temporal memory is represented by the state
`x[t]`.

## Q-to-T Feedback Encoder Options

The Q-to-T models can augment the encoder input with rollout feedback.

With:

```bash
--input-encoder-feedback predicted_temperature
```

the encoder sees:

```text
[u[t], Tin_pred[t]]
```

With:

```bash
--input-encoder-feedback thermal_gaps
```

the encoder sees:

```text
[u[t], Tin_pred[t], Tout[t] - Tin_pred[t], shSetpoint - Tin_pred[t]]
```

Feedback modes require:

```bash
--zero-d
--target-mode absolute
```

`--zero-d` avoids direct feedthrough loops where the output depends
instantaneously on an encoder input that itself depends on the output.

## Q-to-T Direct Feedthrough

By default, the Q-to-T model may learn:

```text
D
```

so inputs can affect `y[t]` instantaneously. With:

```bash
--zero-d
```

the model forces:

```text
D = 0
```

so inputs affect temperature only through the latent state transition.

## Q-to-T Target Modes

The raw model output can be interpreted in three ways.

With:

```bash
--target-mode absolute
```

the model output is the normalized absolute temperature trajectory.

With:

```bash
--target-mode residual
```

the output is anchored to the true initial temperature:

```text
Tin_pred[t] = Tin_initial + raw[t] - raw[0]
```

With:

```bash
--target-mode delta
```

the output is interpreted as a temperature increment and integrated from the
initial temperature.

In current experiments, `absolute` has been the cleanest default.

## Q-to-T Switching Dynamics

Experimental switching dynamics are controlled by:

```bash
--switching-dynamics none
--switching-dynamics heating_bias
--switching-dynamics heating
```

The switching weight is:

```text
alpha[t] = sigmoid(score[t])
```

where the score is based on physical heat input and, when available,
`heat_is_on` and `heat_recently_on`.

With `heating_bias`, off/on systems share:

```text
A, B, C, D, d
```

and switch only the state bias:

```text
b[t] = (1 - alpha[t]) b_off + alpha[t] b_on
```

With `heating`, the model mixes fuller off/on state dynamics. This is more
expressive but easier to destabilize in long rollouts, even if the endpoint
systems are stable.

## Q-to-T Training Loss

### Deterministic Q-to-T Core Loss

The core deterministic objective is temperature MSE:

```text
loss = mean((Tin_pred - Tin_true)^2)
```

### Deterministic Q-to-T Additional Optional Components

- Optional per-window loss normalization:

```bash
--loss-normalization window_std
```

When enabled, predictions and targets are divided by each window's true target
standard deviation before MSE, with floor:

```bash
--loss-std-floor-c
```

- Optional monotonicity regularization:

```bash
--monotonicity-weight
```

Default constrained features:

```text
heat
outdoor_temperature
solar
```

Monotonicity is currently supported only for deterministic Q-to-T without input
encoder and without feedback encoder modes.

## Probabilistic Q-to-T

With:

```bash
--model-kind probabilistic
```

the model samples persistent latent uncertainty per trajectory particle:

```text
xi ~ N(0, I)
theta = theta_net(c, xi)
```

Each particle gets one generated stable state-space system for the whole
rollout. This is trajectory-level persistent model uncertainty.

The probabilistic model uses a memoryless input encoder and can add process
noise:

```bash
--prob-process-noise none
--prob-process-noise constant
--prob-process-noise heteroscedastic
```

Measurement noise is not modeled.

### Probabilistic Q-to-T Core Loss

The core probabilistic objective is:

```text
energy score
```

### Probabilistic Q-to-T Additional Optional Components

Additional weighted terms are available:

```text
variogram score
soft optimistic (soft Best-of-K) trajectory loss
parameter regularization
```

via:

```bash
--prob-variogram-weight
--prob-softopt-weight
--prob-physics-weight
```

The same optional `--loss-normalization window_std` preprocessing can also be
applied before these scores are computed.

Full-profile probabilistic plots show the mean prediction and a 95 percent
scenario interval.

## Q-to-T Evaluation And Plots

Q-to-T logs report temperature metrics in physical units:

```text
RMSE [degC]
MAE  [degC]
NMAE
bias [degC]
```

Saved plots:

```text
test_window_XX_profile_<id>_start_<idx>.html
test_full_profile_XX_<id>.html
```

For probabilistic runs the plot filenames include:

```text
_prob
```

and full-profile plots include a 95 percent scenario interval.

## Q-to-T Assumptions

- The heating input is known at inference time.
- The recommended heating input is `zone_thermal_heating_power`, preferably in
  W/m2.
- The model predicts temperature only.
- It does not predict `Qroom` or HP electric power.
- It does not enforce energy conservation between HP electric power and
  delivered room heat.
- Building metadata determines the state-space parameters and initial-state map.
- The latent state is a learned thermal memory coordinate, not an identified
  physical state.
- Stable `A` is necessary but not sufficient for accurate long rollouts.
- `thermal_gaps` feedback uses static `shSetpoint`, not the timestep thermostat
  setpoint schedule.
- Non-HP buildings can be included when using `zone_thermal_heating_power`
  because the room heat trajectory is already exogenous.
- If `--heating-mode heating_electric` is used, the first input is not room heat.
  That is one reason the closed-loop HP model exists.

# Closed-Loop HP Emulator

The closed-loop model starts where Q-to-T stops. Instead of assuming `Qroom[t]`
is known, it predicts the heating system response from setpoint, weather, and
current predicted room temperature:

```text
setpoint/weather/current room temperature
  -> HP electric power
  -> latent hydronic/buffer energy
  -> delivered room heat
  -> indoor temperature
```

So the thermal state-space block is still present, but its heat input is now
generated by a learned HP/controller/buffer block.

There are now two deterministic closed-loop variants:

```text
closed_loop_hp
  physics-inspired HP/buffer energy balance plus stable thermal SS block

closed_loop_hp_contracting
  data-driven recurrent closed-loop state with guaranteed one-step contraction
```

## What Changes Relative To Q-to-T

Closed-loop models predict three channels instead of one:

```text
[Tin[t+1], Qroom[t], Pel_SH[t]]
```

where:

```text
Tin      = first-zone indoor temperature [degC]
Qroom    = delivered room heat to FL0_THZ0 [W/m2]
Pel_SH   = HP electric power attributed to space heating [W/m2]
```

They require timestep setpoint and HP/DHW columns:

```text
FL0_THZ0, Zone Thermostat Heating Setpoint Temperature
zone_thermal_heating_power
heat_pump_electric_power
hp_mode_is_dhw
hp_ref_capacity_W
hp_size_binding
```

They filter to HP buildings only:

```text
hp_ref_capacity_W is finite and > 0
hp_size_binding == "SH"
```

This excludes DHW-bound HP profiles from all `closed_loop_hp*` pipelines,
because those profiles do not provide a consistent space-heating HP actuator
for the room-heat/electric-power relation.

They require:

```bash
--target-mode absolute
--heat-input-normalization per_floor_area
```

and ignore Q-to-T-specific:

```bash
--input-encoder-feedback
--switching-dynamics
```

because current predicted temperature, outdoor gap, and setpoint gap are always
included internally.

## Closed-Loop Inputs And Targets

The model input is:

```text
u[t] = [
  Tset[t],
  Tout[t],
  solar[t],
  ventilation[t],
  hour_sin[t],
  hour_cos[t],
  day_of_year_sin[t],
  day_of_year_cos[t],
]
```

Alignment is fixed:

```text
u[t] -> Tin[t+1]
u[t] -> Qroom[t]
u[t] -> Pel_SH[t]
```

Targets are constructed as:

```text
Tin_target[t] = FL0_THZ0 zone air temperature at t+1
Qroom[t]      = zone_thermal_heating_power[t] / floor_area
Pel_SH[t]     = heat_pump_electric_power[t] / floor_area
Pel_SH[t]     = 0 when hp_mode_is_dhw[t] > 0.5
```

`Pel_SH` is therefore total HP electric demand with DHW periods masked out, not
necessarily a perfect physical space-heating decomposition.

## Added Heating-System State

In addition to the thermal latent state `x[t]`, the closed-loop model has:

```text
w[t] = bounded latent controller/buffer state
E[t] = nonnegative stored thermal energy [Wh/m2]
```

Default:

```bash
--hp-controller-state-dim 2
--hp-dt-hours 0.25
```

The HP/controller block sees:

```text
[
  u[t],
  Tpred[t],
  Tout[t] - Tpred[t],
  Tset[t] - Tpred[t],
  E[t] / energy_scale,
  w[t],
]
```

It predicts:

```text
pi[t] = HP operation probability
Pel[t] = expected or sampled electric power
COP[t] = linear learnable function of Tout[t]
Qraw[t] = positive requested buffer draw from the room/emitter side
```

COP controls:

```bash
--hp-cop-floor 1.0
--hp-cop-cap 8.0
```

Delivered room heat is now buffer-in-series. HP heat first charges the hydronic
store, the store leaks, and only then can the room draw heat:

```text
Echarged[t]   = clip(E[t] + dt_h * COP[t] * Pel[t], 0, Emax)
Eleaky[t]     = clip((1 - lambda_E[t]) * Echarged[t], 0, Emax)
Qavailable[t] = Eleaky[t] / dt_h
Qroom[t]      = min(Qraw[t], Qavailable[t])
E[t+1]        = clip(Eleaky[t] - dt_h * Qroom[t], 0, Emax)
```

with:

```text
lambda_E[t] = 1 - exp(-loss_rate * dt_h)
```

This removes the old same-step bypass `E[t] / dt_h + COP[t] * Pel[t]`: energy
always passes through the leaky store before becoming delivered room heat.

Physical caps are available:

```bash
--hp-pel-cap-w-m2
--hp-qroom-cap-w-m2
--hp-energy-cap-wh-m2
--hp-energy-cap-hours
--hp-cap-factor
```

Leaving a cap at `0` lets the code derive an automatic cap from the training
data.

## Closed-Loop Thermal Block

The thermal block is the same stable state-space idea as Q-to-T, but its forcing
vector is fixed to:

```text
[
  Tout[t],
  solar[t],
  ventilation[t],
  Qroom[t],
  Tpred[t],
  Tout[t] - Tpred[t],
  Tset[t] - Tpred[t],
]
```

If `--input-encoder-dim` is set, this vector is encoded as:

```text
z[t] = e(...)
```

The update is:

```text
x[t+1] = A x[t] + B z[t] + b
Tpred[t+1] = C x[t+1] + d
```

Closed-loop models force:

```text
D = 0
```

Important caveat: stable `A` only controls the open-loop thermal subsystem.
The full closed-loop map over `[x, w, E, T]` can still expand or become poorly
conditioned.

## Contracting Closed-Loop Variant

With:

```bash
--model-kind closed_loop_hp_contracting
```

the model uses a data-driven bounded recurrent state:

```text
s[t+1] = state_bound * tanh(M_t (s[t] / state_bound) + r_t)
```

where `M_t` and `r_t` are generated from metadata and the exogenous input at
time `t`.

The generated matrix is normalized by Frobenius norm:

```text
||M_t||_F <= contracting_gamma
```

therefore:

```text
||d s[t+1] / d s[t]||_2 <= contracting_gamma
```

for every metadata/input sequence. This is stronger than the sampled Jacobian
penalty used by `closed_loop_hp_probabilistic`.

The HP outputs are still passed through the same buffer-in-series projection as
`closed_loop_hp`: `Pel_SH` charges bounded stored energy, leakage is applied,
and `Qroom` is limited by the remaining available buffer energy. The decoded
`Qraw` is therefore interpreted as requested room/emitter heat, not delivered
heat.

The reported outputs are:

```text
[Tin[t+1], Qroom[t], Pel_SH[t]] = decoder(s[t+1], u[t])
```

`Qroom` and `Pel_SH` are positive and capped by the same data-derived caps used
by the other closed-loop models. Temperature is bounded in normalized units by:

```bash
--contracting-temperature-scale 8.0
```

Main controls:

```bash
--contracting-gamma 0.99
--contracting-state-bound 5.0
--contracting-temperature-scale 8.0
```

The training log reports:

```text
closed_loop_hp_contracting=enabled ... latent_state_guarantee=||dF/ds||_2<=gamma
rho_max=<contracting_gamma>
```

## Closed-Loop Deterministic Loss

### Core Loss

The core deterministic closed-loop objective is MSE over the three normalized
channels:

```text
prediction = [Tin_pred, Qroom_pred, Pel_SH_pred]
target     = [Tin_true, Qroom_true, Pel_SH_true]
loss       = mean((prediction - target)^2)
```

### Additional Optional Components

An auxiliary HP on/off BCE can be added:

```text
hp_on_true[t] = Pel_SH_true[t] > heat_on_threshold
```

with weight:

```bash
--hp-mode-loss-weight 0.1
```

## Probabilistic Closed-Loop Additions

With:

```bash
--model-kind closed_loop_hp_probabilistic
```

the original probabilistic closed-loop model samples persistent:

```text
xi ~ N(0, I)
```

This `xi` conditions:

```text
A, B, C, b, d
x0
w0
E0
COP parameters
buffer loss behavior
```

With:

```bash
--model-kind closed_loop_hp_contracting_probabilistic
```

the same probabilistic objective is used, but the sampled thermal model is the
bounded contractive architecture:

```text
s[t+1] = state_bound * tanh(M_t (s[t] / state_bound) + r_t + noise[t])
||M_t||_F <= contracting_gamma
```

Here `xi` conditions the transition, initial state, initial stored energy, HP
emission, COP, and buffer loss parameters. HP heat still passes through the
buffer-in-series projection before becoming `Qroom`.

HP electric power is modeled as:

```text
m[t] ~ Bernoulli(pi[t])
log1p(Pel_active[t]) ~ Normal(mu[t], sigma[t])
Pel[t] = m[t] * Pel_active[t]
```

During training, the thermal and buffer path uses expected HP power for stable
differentiability:

```text
E[Pel[t]] = pi[t] * (exp(mu[t] + 0.5 sigma[t]^2) - 1)
```

Scenario plots use:

```bash
--prob-hp-scenario-mode bernoulli
```

by default, so sampled `Tin`, `Qroom`, and `Pel_SH` traces stay linked within
each scenario.

### Probabilistic Closed-Loop Core Loss Components

For `closed_loop_hp_probabilistic` and
`closed_loop_hp_contracting_probabilistic`, the core objective combines:

```text
energy score over [Tin, Qroom, Pel_SH]
HP on/off BCE
active HP electric log-power NLL
```

### Probabilistic Closed-Loop Additional Optional Components

Additional weighted terms are:

```text
variogram score
soft Best-of-K trajectory loss
inactive HP leakage penalty
physics regularization
sampled closed-loop stability penalty
```

In training code, all probabilistic closed-loop losses are computed with
`hp_scenario_mode="expected"` (differentiable expected electric power path),
while scenario plots can still use Bernoulli sampling.

The energy score is the proper-scoring-rule component. Once auxiliary terms are
added, the total objective is no longer strictly proper.

## Closed-Loop Stability Penalty

An optional sampled local contraction penalty is available for both
probabilistic closed-loop variants:

```bash
--closed-loop-stability-weight
--closed-loop-stability-gamma
--closed-loop-stability-samples
--closed-loop-stability-aggregation mean|max
```

It samples batch/particle/time states and penalizes:

```text
relu(||dF/ds||_2 - gamma)^2
```

where:

```text
s = [x, w, E, T]
```

This is only a local sampled regularizer. It is not a formal global stability
proof.

## Closed-Loop Warm Start

The recommended probabilistic closed-loop workflow is:

1. Train and save a deterministic `closed_loop_hp` model.
2. Warm-start `closed_loop_hp_probabilistic` from that deterministic artifact.

Warm-start command option:

```bash
--init-from-deterministic-artifact output/neural_building_emulator/model_checkpoints/closed_loop_hp/selected
```

The artifact must have:

```text
model_kind = closed_loop_hp
```

The warm start copies the deterministic solution but now adds small nonzero
initial sensitivity to `xi`:

```bash
--init-xi-weight-scale 0.05
```

and initializes active HP log-power uncertainty with:

```bash
--init-hp-active-log-sigma 0.25
```

Set `--init-xi-weight-scale 0` to recover the old exactly xi-blind deterministic
copy.

## Shared Artifact Layout

Saved models are stored under model-kind-specific directories:

```text
output/neural_building_emulator/model_checkpoints/deterministic/selected
output/neural_building_emulator/model_checkpoints/probabilistic/selected
output/neural_building_emulator/model_checkpoints/closed_loop_hp/selected
output/neural_building_emulator/model_checkpoints/closed_loop_hp_probabilistic/selected
```

This is important for warm starts: a probabilistic save should not overwrite the
deterministic artifact it was initialized from.

Artifact files are suffixed by model kind:

```text
model_<model_kind>.eqx
scalers_<model_kind>.npz
metadata_<model_kind>.json
```

## Closed-Loop Evaluation And Plots

Closed-loop logs report physical-unit RMSE for:

```text
Tin      [degC]
Qroom    [W/m2]
Pel_SH   [W/m2]
```

Closed-loop plots have three stacked panels:

```text
Tin
Qroom
Pel_SH
```

Deterministic plot filenames:

```text
test_window_XX_profile_<id>_start_<idx>_closed_loop_hp.html
test_full_profile_XX_<id>_closed_loop_hp.html
```

Probabilistic plot filenames:

```text
test_window_XX_profile_<id>_start_<idx>_closed_loop_hp_prob.html
test_full_profile_XX_<id>_closed_loop_hp_prob.html
```

For the probabilistic closed-loop model, the HTML plots show scenario mean and a
95 percent interval. Current epoch metrics still mostly select by `test_rmse_c`,
so they can prefer narrow deterministic-like checkpoints.

## Closed-Loop Known Issues

- Deterministic closed-loop training is currently the strongest closed-loop
  sanity baseline.
- The probabilistic closed-loop model can fit mean trajectories after warm-start,
  but temperature intervals can remain too narrow.
- Stable thermal `A` does not imply full closed-loop stability.
- `closed_loop_hp_contracting` guarantees recurrent-state contraction, but gives
  up the explicit HP energy-conservation equation.
- The Jacobian penalty is local and sampled, not a proof.
- Training uses expected HP power in the thermal path; Bernoulli HP sampling is
  used for scenarios/plots.
- The active HP power NLL can dominate the loss if initialized too narrowly or if
  `Pel_SH` has sharp spikes.
- `Pel_SH` depends on the quality of `hp_mode_is_dhw`.
- The latent energy store is physically motivated but not directly observed.
- Epoch logs do not yet report interval coverage or mean 95 percent width.

# Practical Commands

## Deterministic Q-to-T

```bash
.venv/bin/python -m neural_building_emulator.train \
  --model-kind deterministic \
  --heating-mode zone_thermal \
  --heat-input-normalization per_floor_area \
  --max-profiles 1000 \
  --test-fraction 0.1 \
  --epochs 20 \
  --batch-size 32 \
  --sequence-length 960 \
  --stride 2024 \
  --state-dim 5 \
  --hidden-dim 32 \
  --input-encoder-dim 8 \
  --input-encoder-hidden-dim 64 \
  --input-encoder-depth 2 \
  --schur-mode near_identity \
  --schur-gamma 0.99 \
  --target-mode absolute \
  --zero-d \
  --input-encoder-feedback thermal_gaps \
  --num-full-profile-plots 5 \
  --save-model
```

## Probabilistic Q-to-T

```bash
.venv/bin/python -m neural_building_emulator.train \
  --model-kind probabilistic \
  --heating-mode zone_thermal \
  --heat-input-normalization per_floor_area \
  --max-profiles 1000 \
  --test-fraction 0.1 \
  --epochs 20 \
  --batch-size 32 \
  --sequence-length 960 \
  --stride 2024 \
  --state-dim 5 \
  --hidden-dim 32 \
  --input-encoder-dim 8 \
  --input-encoder-hidden-dim 64 \
  --input-encoder-depth 2 \
  --schur-mode near_identity \
  --schur-gamma 0.99 \
  --target-mode absolute \
  --zero-d \
  --input-encoder-feedback thermal_gaps \
  --prob-particles 10 \
  --prob-eval-particles 16 \
  --prob-latent-dim 4 \
  --prob-process-noise heteroscedastic \
  --num-full-profile-plots 5 \
  --save-model
```

## Deterministic Closed-Loop HP

```bash
.venv/bin/python -m neural_building_emulator.train \
  --max-profiles 1000 \
  --test-fraction 0.1 \
  --epochs 20 \
  --batch-size 32 \
  --sequence-length 960 \
  --stride 2024 \
  --state-dim 5 \
  --hidden-dim 32 \
  --input-encoder-dim 8 \
  --input-encoder-hidden-dim 64 \
  --input-encoder-depth 2 \
  --schur-mode near_identity \
  --schur-gamma 0.99 \
  --target-mode absolute \
  --num-full-profile-plots 5 \
  --model-kind closed_loop_hp \
  --save-model
```

## Contracting Closed-Loop HP

```bash
.venv/bin/python -m neural_building_emulator.train \
  --max-profiles 1000 \
  --test-fraction 0.1 \
  --epochs 20 \
  --batch-size 32 \
  --sequence-length 960 \
  --stride 2024 \
  --state-dim 5 \
  --hidden-dim 32 \
  --input-encoder-dim 8 \
  --input-encoder-hidden-dim 64 \
  --input-encoder-depth 2 \
  --target-mode absolute \
  --num-full-profile-plots 5 \
  --model-kind closed_loop_hp_contracting \
  --contracting-gamma 0.99 \
  --contracting-state-bound 5.0 \
  --contracting-temperature-scale 8.0 \
  --save-model
```

## Probabilistic Closed-Loop HP Warm Start

```bash
.venv/bin/python -m neural_building_emulator.train \
  --max-profiles 1000 \
  --test-fraction 0.1 \
  --epochs 20 \
  --batch-size 32 \
  --sequence-length 960 \
  --stride 2024 \
  --state-dim 5 \
  --hidden-dim 32 \
  --input-encoder-dim 8 \
  --input-encoder-hidden-dim 64 \
  --input-encoder-depth 2 \
  --schur-mode near_identity \
  --schur-gamma 0.99 \
  --target-mode absolute \
  --num-full-profile-plots 5 \
  --model-kind closed_loop_hp_probabilistic \
  --prob-particles 10 \
  --prob-eval-particles 16 \
  --prob-plot-particles 100 \
  --prob-latent-dim 4 \
  --prob-process-noise heteroscedastic \
  --prob-hp-scenario-mode bernoulli \
  --closed-loop-stability-weight 0.05 \
  --closed-loop-stability-gamma 0.995 \
  --init-from-deterministic-artifact output/neural_building_emulator/model_checkpoints/closed_loop_hp/selected \
  --init-xi-weight-scale 0.05 \
  --init-hp-active-log-sigma 0.25 \
  --save-model
```
