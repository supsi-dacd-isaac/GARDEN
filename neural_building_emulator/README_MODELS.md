# Neural Building Emulator Models

The building-local causal semi-Markov research model and its half-year
ablation are documented separately in
`[README_CAUSAL_HYBRID_HP.md](README_CAUSAL_HYBRID_HP.md)`.

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

`totalFloors` is therefore an explicit network input. Closed-loop HP models
extend this common list with the equipment metadata described below.

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

For the non-recommended `--heating-mode heating_electric` case, the source is
whole-building HP power. New runs therefore divide it by
`floor_area * totalFloors`; `--hp-power-area-normalization zone_floor_area`
retains the historical behavior for artifact reproduction.

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
Pel_SH   = whole-building HP electric power attributed to space heating [W/m2]
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
  space_heating_available[t],
  hour_sin[t],
  hour_cos[t],
  day_of_year_sin[t],
  day_of_year_cos[t],
]
```

Closed-loop HP models additionally receive these static equipment features:

```text
hp_ref_capacity_W / (floor_area * totalFloors)
SH_design_cap_W / (floor_area * totalFloors)
shVolume_m3 / (floor_area * totalFloors)
hp_ref_cop
one-hot(hp_model_name): aerotop_g07_14m, aerotop_t35r
```

The resulting metadata vector has 21 entries. These equipment values are
standardized with the other metadata using training-set statistics.

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
Pel_SH[t]     = heat_pump_electric_power[t] / (floor_area * totalFloors)
Pel_SH[t]     = 0 when hp_mode_is_dhw[t] > 0.5
Qroom[t] = Pel_SH[t] = 0 from May 15 through September 30
```

The two power targets have different physical scopes. `Qroom` is the heat
delivered to the modeled first-zone apartment, so its denominator is that
zone's `floor_area`. `heat_pump_electric_power` is the total HP electricity for
the building, so its denominator is the estimated total heated area
`floor_area * totalFloors`. This removes a spurious dependence of electric-power
intensity and flexibility KPIs on the number of floors.

New training uses `--hp-power-area-normalization building_heated_area` by
default. `--hp-power-area-normalization zone_floor_area` is retained only to
reproduce artifacts trained before this correction. Saved artifacts record the
mode, and artifact-based plotting and KPI scripts automatically use the saved
value. Artifacts without the field are interpreted as legacy
`zone_floor_area` artifacts.

The buffer equations therefore relate a first-zone heat intensity to a
whole-building-average electric intensity. This is a useful approximation for
the present dataset, but it is not an exact whole-building energy balance. An
exact balance would additionally require total heat delivered to all zones.

`space_heating_available` exactly reproduces the EnergyPlus seasonal SH
lockout. It hard-gates predicted room heat and SH electric power, while the
thermal transition can react only through `Qroom`. Electricity during the
summer lockout is also removed from the SH target, covering DHW events missed
by `hp_mode_is_dhw`.

Training window grids rotate across one stride over the epochs by default;
validation starts remain fixed and the per-epoch window count is unchanged.
`--no-rotate-window-starts` restores the old fixed start grid.

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
s[t+1] = state_bound * tanh(M_t(c, f_t) (s[t] / state_bound) + b_t(c, f_t))
```

Here `f_t` is the current thermal forcing. One joint transition network emits
both `M_t` and `b_t` from `[c, f_t]` at every timestep, restoring the
architecture used before the fixed-matrix experiment.

The transition generator can optionally be restricted to exogenous forcing:

```bash
--contracting-transition-conditioning exogenous
```

The default `state_feedback` retains the historical conditioning. With
`exogenous`, the `Qroom`, predicted `Tin`, and `Tout-Tin` slots are zeroed only
before the joint network that emits `M_t` and `b_t`. The parameter-tree shape
is unchanged, and the HP controller plus the explicit positive Q-to-T path are
unaffected. Consequently,

```text
(M_t, b_t) = g(c, xi, u_exogenous[t])
```

for the probabilistic model (without `xi` for the deterministic model). This
removes the `dM_t/dTin` and `db_t/dTin` feedback paths. It does not by itself
prove contraction of the complete stochastic closed loop: in heteroscedastic
mode the process-noise scale still receives the latent state, and the HP,
buffer, Q response, and temperature states remain coupled.

The factorized alternative keeps the matrix generator exogenous while restoring
temperature feedback through a separate additive state-space input path:

```bash
--contracting-transition-conditioning exogenous_additive_feedback \
--contracting-additive-feedback-gain-bound 1.0
```

Its transition is

```text
(M_t, B_t, b_t) = g(c, xi, u_exogenous[t])
z_t = tanh(e([u_exogenous[t], Tin[t], Tout[t] - Tin[t]]))
s[t+1] = state_bound * tanh(
    M_t (s[t] / state_bound) + B_t z_t + b_t + process_noise[t]
)
```

There is no `xi` or process-noise term in the deterministic model. `Tset` and
space-heating availability do not enter this thermal forcing; they act through
the HP/controller path. With `positive_leaky`, `Qroom` is also excluded from
`z_t` because it enters temperature through the separate positive leaky modes.
With `unconstrained`, `Qroom` remains part of `z_t`.

Both generated matrices are pointwise bounded:

```text
||M_t||_F <= contracting_gamma
||B_t||_F <= contracting_additive_feedback_gain_bound
```

Compared with `state_feedback`, this removes the recurrently amplified
`(dM_t/dTin) s[t]` term while retaining the information lost by `exogenous`.
It remains an empirical closed-loop architecture rather than a contraction
proof because the encoded forcing still depends on predicted `Tin`.

The generated matrix is normalized by Frobenius norm:

```text
||M_t(c, f_t)||_F <= contracting_gamma
```

This guarantees `||M_t||_2 <= contracting_gamma` pointwise. It does not by
itself bound the complete HP-buffer-temperature Jacobian. In the default
`unconstrained` Q-to-T mode, both `M_t` and `b_t` depend on predicted `Qroom`
and `Tin`; the optional sampled whole-state penalty handles that feedback
empirically.

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



### Structural Positive Q-to-T Path

Contracting deterministic and probabilistic models can instead use:

```bash
--contracting-temperature-update leaky_equilibrium
--contracting-temperature-delta-max-c 2.0
--contracting-q-to-t-mode positive_leaky
--contracting-q-to-t-time-constants-hours 0.25 1 4 16 24 48
```

This removes `Qroom` from the neural inputs generating `M_t` and `b_t`. A
separate positive leaky state carries delivered heat:

```text
g_total(c[, xi]) in [g_min, g_max]
w_j(c[, xi]) >= 0, sum_j w_j = 1
lambda_j = exp(-dt / tau_j)
h_j[t+1] = lambda_j h_j[t]
             + (1 - lambda_j) w_j g_total Qroom[t]
T_eq[t] = clip(T_eq,free[t] + sum_j h_j[t+1], -Tbound, Tbound)
Tin[t+1] = (1 - alpha) Tin[t] + alpha T_eq[t]
```

The total gain has physical units `degC/(W/m2)` and is bounded with:

```bash
--contracting-q-to-t-gain-min-c-per-w-m2 0.01
--contracting-q-to-t-gain-max-c-per-w-m2 2.0
```

For fixed metadata, particle and current state, increasing delivered room heat
cannot lower the next equilibrium temperature through the generated matrix or
bias. Saturation can make the response flat at the configured temperature
bound, but cannot reverse its sign. In the probabilistic model the gains and
mode weights are conditioned on the trajectory-persistent `xi`, and the leaky
heat states are included in sampled full closed-loop Jacobian diagnostics.

The learned outer `alpha` can be removed with the optional bounded-equilibrium
readout:

```bash
--contracting-temperature-update bounded_equilibrium
--contracting-temperature-delta-max-c 2.0
```

It keeps the same equilibrium and positive leaky Qroom response, but updates
temperature as:

```text
delta_max_scaled = delta_max_C / temperature_target_scale
Tin[t+1] = clip(
    Tin[t] + delta_max_scaled * tanh((T_eq[t] - Tin[t]) / delta_max_scaled),
    -Tbound,
    Tbound,
)
```

There is no learned `alpha` in this mode. For small equilibrium errors the
update approaches `T_eq`; for large errors its physical magnitude is strictly
bounded by `--contracting-temperature-delta-max-c`. This simplifies the
readout and avoids the additional slow pole introduced by a small learned
`alpha`. It remains an explosion guard, not a proof that the full nonlinear
HP-buffer-thermal system is globally contractive.

A second optional mode keeps a learned local response rate:

```bash
--contracting-temperature-update alpha_bounded_equilibrium
--contracting-temperature-delta-max-c 2.0
```

```text
alpha = sigmoid(alpha_net(c[, xi]))
Tin[t+1] = clip(
    Tin[t] + delta_max_scaled
        * tanh(alpha * (T_eq[t] - Tin[t]) / delta_max_scaled),
    -Tbound,
    Tbound,
)
```

For deterministic models, `alpha(c)` is fixed for a building and the entire
rollout. For probabilistic models, `alpha(c, xi)` is fixed within a particle but
can differ across particles. Near equilibrium the update is approximately
`alpha * (T_eq - Tin)`, so alpha controls the local response speed. The outer
`tanh` independently retains the hard physical step cap, allowing alpha to use
the full `(0,1)` range instead of the much smaller `alpha_max` required by the
linear `leaky_equilibrium` update.

Main controls:

```bash
--contracting-gamma 0.99
--contracting-state-bound 5.0
--contracting-temperature-scale 8.0
```

The training log reports:

```text
closed_loop_hp_contracting=enabled ... thermal_matrix=time_varying pointwise_matrix_guarantee=||M_t||_2<=gamma
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

The pointwise channel weights can be changed with:

```bash
--closed-loop-trajectory-output-weights 1 1 1
```

in `Tin Qroom Pel_SH` order. The weights are normalized to mean one, so changing
them changes the channel tradeoff without changing the nominal loss scale. For
example, `1 1 0` removes direct pointwise `Pel_SH` MSE while retaining the HP
mode BCE and the indirect `Pel -> buffer -> Qroom -> Tin` path.

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
s[t+1] = state_bound * tanh(M_t(c, xi, f_t) (s[t] / state_bound) + b_t(c, xi, f_t) + noise[t])
||M_t(c, xi, f_t)||_F <= contracting_gamma
```

Here `xi` remains fixed over the particle trajectory, while `M_t` is regenerated
from the current thermal forcing at every step. A joint transition network emits
both `M_t` and `b_t`. `xi` also conditions the nonlinear forcing, initial state, initial stored energy, HP
emission, COP, and buffer loss parameters. HP heat still passes through the
buffer-in-series projection before becoming `Qroom`.

HP electric power is modeled as:

```text
m[t] ~ Bernoulli(pi[t])
log1p(Pel_active[t]) ~ Normal(mu[t], sigma[t])
Pel[t] = m[t] * Pel_active[t]
```

By default, the thermal and buffer path uses expected HP power for stable
differentiability:

```text
E[Pel[t]] = pi[t] * E[expm1(clip(X[t], 0, log1p(Pmax)))]
X[t] ~ Normal(mu[t], sigma[t]^2)
```

The bounded expectation is evaluated analytically and therefore matches the
clipped active-power distribution used by scenario sampling. It includes both
the log-normal variance correction and probability mass clipped at zero and at
`Pmax`.

Stochastic HP propagation can instead be enabled during training with:

```bash
--prob-hp-training-mode straight_through
--prob-hp-concrete-temperature 0.5
```

Each training particle then samples active-power noise and a hard HP on/off
trace. The forward mode is binary, while its backward derivative uses a Binary
Concrete relaxation. Thus the energy, variogram, IRES, and soft Best-of-K terms
see linked sampled `Pel -> E -> Qroom -> Tin` trajectories. The BCE and active
power NLL remain explicit auxiliary losses. The default
`--prob-hp-training-mode expected` preserves the previous training behavior.

### Direct flexibility-KPI CRPS

An optional coefficient-level score targets the same upward and downward
event-study gains used by `flexibility_event_study_kpis.py`:

```bash
--prob-flex-kpi-crps-weight 1.0
--prob-flex-kpi-crps-horizons-hours 0.5 1 2 3
--prob-flex-kpi-crps-setpoint-threshold-c 0.05
--prob-flex-kpi-crps-min-events 20
--prob-flex-kpi-crps-ridge 0.001
--prob-flex-kpi-crps-controls full
```

For each training window and horizon, the model forms the pre/post HP-power
responses and fits one fixed ridge-regression design:

```text
delta_P_H = alpha + beta_plus delta_Tset_plus
                  + beta_minus delta_Tset_minus + controls
G_plus = H beta_plus
G_minus = H beta_minus
```

The design uses observed setpoints and disturbances. With `full` controls it
also uses observed pre-event power and temperature, previous setpoint, and
previous setpoint jump. It is fixed across particles, so every particle
coefficient is only a linear projection of its predicted `Pel` trace. The loss
is empirical univariate CRPS averaged over `G_plus`, `G_minus`, horizons, and
eligible profile windows:

```text
CRPS = mean_k |G_k - G_true| - 0.5 mean_kl |G_k - G_l|
```

The coefficients are fitted in normalized training units; conversion to
Wh/(m2 K) is a constant scaler ratio and therefore does not change the
calibration target. Ridge regularization and a minimum-event mask stabilize
short-window regressions. Startup diagnostics report the design dimension,
effective event requirement, and eligible training windows per horizon.
Existing behavior is unchanged when the weight is zero, which is the default.

The contracting probabilistic model also has an optional persistent HP
controller:

```bash
--prob-hp-activation-model persistent_markov
--prob-hp-persistent-latent-dim 2
--prob-hp-controller-leak 0.25
--prob-hp-controller-noise-scale 0.0
```

For each particle it samples a dedicated `r_HP ~ Normal(0, I)` once for the
whole trajectory and carries a bounded controller state `h[t]`. Separate
transition heads produce:

```text
p_start[t] = sigmoid(g_start(c, xi, r_HP, h[t], u[t], T[t], E[t]))
p_stop[t]  = sigmoid(g_stop (c, xi, r_HP, h[t], u[t], T[t], E[t]))
Pr(m[t]=1) = (1-m[t-1]) p_start[t] + m[t-1] (1-p_stop[t])
h[t+1]     = (1-leak) h[t] + leak * tanh(f(..., m[t]) + noise[t])
```

This makes complete on/off spells coherent within a scenario instead of
sampling unrelated Bernoulli states at every timestep. `r_HP` also conditions
active power, requested room heat, COP, and buffer-loss parameters. During
training, `--hp-mode-loss-weight` weights a teacher-forced transition NLL:
`p_start` is scored on reference-off steps and `p_stop` on reference-on steps.
The probabilities are averaged across particles before this likelihood is
evaluated, so the persistent latent is not forced to make every particle
identical.

`independent` remains the default and preserves the previous HP mode
architecture. The persistent controller is currently implemented only for
`closed_loop_hp_contracting_probabilistic`. Its bounded controller and
previous-mode probability are included in the sampled whole-state Jacobian
penalty.

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

The same option:

```bash
--closed-loop-trajectory-output-weights 1 1 0
```

removes `Pel_SH` from the trajectory energy score and soft Best-of-K distance.
It does not disable HP mode BCE, active-power NLL, inactive leakage, IRES, or
flexibility-coefficient CRPS. The default `1 1 1` preserves the historical
loss exactly.

### Probabilistic Closed-Loop Additional Optional Components

Additional weighted terms are:

```text
variogram score
soft Best-of-K trajectory loss
inactive HP leakage penalty
physics regularization
sampled closed-loop stability penalty
```

The training HP path is selected independently from plot generation:

```text
--prob-hp-training-mode expected|straight_through
--prob-hp-scenario-mode expected|bernoulli
--prob-hp-activation-model independent|persistent_markov
```

The straight-through gradient is biased, even though its forward trajectories
contain hard Bernoulli decisions. The energy score remains proper as a function
of the sampled predictive trajectories; adding auxiliary losses means the full
combined objective is not itself a strictly proper score.

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
--closed-loop-stability-every-steps
--closed-loop-stability-method exact_svd|power_iteration
--closed-loop-stability-power-iterations
```

It samples batch/particle/time states and penalizes:

```text
relu(||dF/ds||_2 - gamma)^2
```

where:

```text
s = active recurrent state
```

The active state contains `[x, E, T]` plus the controller/previous-mode state
for `persistent_markov`, or the power/mode histories for `power_history`.
Dormant identity states are excluded. Heteroscedastic-noise checks use sampled
nonzero noise. The default differentiable power iteration is evaluated only
once every configured number of optimizer steps; `exact_svd` remains available
for diagnostics. HP activation uses the differentiable expected transition
because hard Bernoulli switching has no useful classical Jacobian at its
boundary. This is still a local sampled regularizer, not a formal global
stability proof, and finite power iteration can underestimate the largest
singular value.

Fixed-matrix contracting artifacts have an incompatible parameter tree with the
restored time-varying production model and must be retrained.

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
- `closed_loop_hp_contracting` structurally bounds every generated thermal
matrix and all reported states/outputs, but does not prove contraction of the
complete nonlinear HP feedback map.
- The Jacobian penalty is local and sampled, not a proof.
- Expected HP training remains the default. Straight-through stochastic HP
training is optional and may have higher minibatch gradient variance.
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



### Structurally monotone thermostat demand

Both contracting closed-loop models support:

```text
--hp-thermostat-demand-mode monotone
```

In this mode the timestep thermostat setpoint is masked from the unrestricted HP
input encoder. The model instead uses bounded, context-conditioned positive
slopes and thresholds so latent requested room heat is nondecreasing in
`Tset - Tin` when metadata, weather, particle latent and current indoor temperature are held fixed. HP activation and active power remain unrestricted functions of the thermal gap, controller history and buffer energy; this allows compressor cycling and delayed buffer chartrging. Delivered room heat is the requested heat capped by available buffer energy, so it is not independently forced to be pointwise monotone. The default `unconstrained` mode preserves old
artifacts and behavior. Optional bounds are exposed through
`--hp-thermostat-slope-{min,max}` and
`--hp-thermostat-threshold-{min,max}-c`.

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
