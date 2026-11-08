# Neural building emulators: current models and evidence

Updated **2026-11-06**. This README describes the completed v9-final comparisons,
their actual selected models, and the limits of the evidence. Historical options
in the code are not necessarily active in these experiments.

## 1. Scope

The goal is to generate reliable annual building trajectories and distributions
of flexibility KPIs from metadata, weather, loads and prescribed control policies.
The emulator should require a small set of generic modeling assumptions and no
auxiliary EnergyPlus simulation for each new prediction. Reproducing the simulator's
internal component/controller implementation is not the objective.

Two distinct experiments address different parts of that goal:

| Experiment | Information supplied at inference | Predictions | Question |
| --- | --- | --- | --- |
| Thermal-only | Observed delivered room heat and ventilation, weather, gains, occupancy, metadata and initial indoor temperature | `Tin` | What do dynamical structure, contraction and noise conditioning contribute to thermal prediction? |
| Full emulator | Setpoints, requested force-off, seasonal availability, weather, gains, occupancy, DHW withdrawal, metadata and initial indoor temperature | `Tin`, `Qroom`, `Pel_SH`; two auxiliary tank temperatures in one model | Can an autonomous emulator reproduce thermal/HP behavior and flexibility under changed control policies? |

Annual inference is sequential: generated states and outputs affect later
predictions. True targets are not repeatedly fed back during annual scoring.
Persistent latent draws are held through the year; fresh process innovations are
sampled through time. Training uses shorter windows initialized at their observed
starting temperature, which is a different task from uninterrupted annual scoring.

Native trajectory uncertainty is the primary result. RNS and RNS–ECC are separate
KPI-level post-processing comparisons: they do not change or calibrate physical
time traces. Good pointwise coverage, good annual KPI calibration and numerical
rollout stability are evaluated separately.

The two reference runs are:

- **Thermal-only:** [ticino_v9_thermal_four_models_1000_e30](../output/neural_building_emulator_refactor/ticino_v9_thermal_four_models_1000_e30),
four models, shared validation stopping between 15 and 30 epochs.
- **Full emulator:** [ticino_v9_full_benchmark_1000_fixed_e15](../output/neural_building_emulator_refactor/ticino_v9_full_benchmark_1000_fixed_e15),
four neural models with a 15-epoch budget, plus a metadata-only LightGBM KPI
benchmark. This supersedes the earlier adaptive run that stopped the two
structured models at 10 epochs.

These are different tasks and training objectives. Their absolute temperature
scores should not be interpreted as a controlled comparison against each other.

## 2. Dataset description

The current campaign is
`data/garden-ticino-five-cities-1000-v9-final`. Its
[manifest](../data/garden-ticino-five-cities-1000-v9-final/campaign_manifest.json) records **1,000 buildings × four policies = 4,000
annual simulations**. Buildings were sampled from five Ticino municipalities
(BFS codes 5192, 5002, 5113, 5254 and 5250). Weather is shared within each
municipality. The campaign assigns heat pumps counterfactually to all buildings;
it is not a sample restricted to buildings with observed real-world heat pumps.

Each building/policy directory contains:

```text
data/control=<policy>/egid=<building>/
    timeseries.parquet
    simulation_metadata.json
    complete.json
```

The simulations cover **2023 at 15-minute resolution**, in local standard time
without daylight-saving shifts: 35,040 recorded intervals. Labels identify
interval starts, while indoor temperatures represent interval-end states.
The adapter uses the first recorded temperature for initialization and produces
35,039 aligned prediction steps. Power and interval-mean tank-temperature signals
must not be treated as instantaneous endpoint measurements.

### Policies and folds

| Dataset | Prescribed intervention | Use |
| --- | --- | --- |
| `setpoint_excitation_v2` | Random setpoint offsets within ±2°C; multiple dwell lengths; no requested HP force-off | Training, validation and source-policy test |
| `force_off_excitation_v2` | Fixed nominal setpoint; extended randomized HP force-off events | Training, validation and source-policy test |
| `tariff_setpoint_v2` | Price-dependent setpoint shifts, with randomized active-block magnitudes of 1–2°C | Held-out policy evaluation |
| `tariff_force_off_v2` | Price-dependent requested HP force-off at the nominal setpoint | Held-out policy evaluation |

The tariff policies are **separate intervention datasets**, not one policy that
simultaneously changes both controls. Their schedules are prepared in advance;
they are not comfort-aware feedback controllers driven by simulated `Tin`.
The v9 force-off excitation revision specifies 2, 4, 6, 8 and 12-hour interruptions,
6/8/12-hour recovery choices and a 16-hour rolling limit. The
[campaign configuration](../data/garden-ticino-five-cities-1000-v9-final/campaign.yaml) and retained schedules are the authority
for exact control rules; older dataset-generation proposals are not the executed
v9 specification.

Force-off disables generation through the producer's final interlock, including
the specified SH/DHW equipment. **Stored energy and delivered zone heat are not
directly zeroed.** Continued room heating during an interruption is therefore
possible; the emulator must represent depletion and subsequent recovery.

The [building split](../data/garden-ticino-five-cities-1000-v9-final/split_manifest.json) is preserved across policies and experiments:

| Fold | Buildings | Use |
| --- | ---: | --- |
| Training | 701 | Model fitting; excitation policies only |
| Validation | 100 | Checkpoint selection; excitation policies only |
| Calibration | 99 | Reserved and unused in thermal-only; source-policy RNS fitting in the full comparison |
| Test | 100 | Frozen evaluation under all four policies |

The split seed is `20261028`; these model comparisons use training seed `13`.
Repeated policies do not create additional independent buildings. No tariff data
are used to train models, select checkpoints or fit recalibration. The final
package excludes failed EGID 400007253 and includes replacement EGID 190191859,
with the preserved final fold membership.

### Signals, normalization and information availability

- `Tin`: indoor temperature of zone `FL0_THZ0`, in °C.
- `Qroom`: delivered heat to that zone, normalized by its `floor_area`, in W/m².
- `Pel_SH`: HP electricity attributed to space heating under the dataset's DHW
masking convention, normalized by `floor_area × totalFloors`, in W/m². It is
not total building electricity or total SH+DHW electricity.
- Disturbances: outdoor temperature, solar radiation, non-people internal gains,
occupancy, DHW withdrawal where used, and calendar encodings.
- Commands: heating setpoint, requested force-off and space-heating availability.
Effective force-off and realized plant states are diagnostic signals, not
replacements for the commanded inputs.
- Metadata: building type, geometry, age/renovation descriptors, thermal-mass
class, dwelling count, HP capacity/COP/model, SH design capacity and SH volume.
All four full models additionally receive DHW volume and recovery capacity.
- Auxiliary labels: `sh_tank_measured_temperature_c` maps to
`sh_tank_mean_temperature_c`; `dhw_tank_mean_temperature_c` is the DHW label.
Both are reporting-interval means. They supervise the two-tank model but are
not measured tank-temperature inputs at inference.

The thermal-only experiment supplies **recorded ventilation**. Full-model annual
evaluation instead uses `ventilation_rollout_mode=eplus_rule`, applying the
implemented ventilation rule to each particle's predicted indoor temperature.
This distinction is essential when interpreting transferability.

The current adapters are [dataset_adapter.py](../neural_building_emulator/dataset_adapter.py),
[data.py](../neural_building_emulator/data.py) and
[columns.py](../neural_building_emulator/columns.py). Model scalers are fitted on
training data only. Campaign hashes were checked for paired physics and exogenous
inputs; this is a provenance audit, not an independent reconstruction of E+ inputs.

## 3. The four thermal-only models

Implementation: [models/thermal_noise.py](models/thermal_noise.py).
Runner: [thermal_noise_experiment.py](thermal_noise_experiment.py).
The [frozen protocol](../output/neural_building_emulator_refactor/ticino_v9_thermal_four_models_1000_e30/comparison_request.json) records the executed settings.
There is **no Schur arm in this comparison**: the structured transition is diagonal.

For each building, let `m` be standardized metadata, `u_t` the prescribed inputs
(including observed `Qroom` and ventilation), and
\(\xi\sim\mathcal N(0,I_4)\) a trajectory-persistent latent. The structured update,
in normalized coordinates, is

\[
h_{t+1}=\operatorname{diag}(\rho(m,\xi))h_t
        +B(m,\xi)\phi(u_t)+b(m,\xi)
        +\sigma(m,\xi,u_t,h_t)\odot\varepsilon_t,
\qquad
\widehat T_{t+1}=C(m,\xi)h_{t+1}+d(m,\xi),
\]

with independent standard Gaussian innovations `epsilon_t`. The forcing encoder
has no generated-temperature/state feedback. There is no separate direct
feedthrough term in the temperature readout, and no clipping of predicted `Tin`.

| Variant ID | Dynamics and noise | State size | Trainable parameters |
| --- | --- | ---: | ---: |
| `diagonal_exogenous` | Constrained positive diagonal poles; diffusion depends on metadata, latent and prescribed inputs only | 8 | 20,554 |
| `diagonal_unconstrained_exogenous` | Same structure and exogenous diffusion; positive poles can exceed one | 8 | 20,554 |
| `diagonal_state` | Same constrained deterministic core; diffusion also sees the current hidden state | 8 | 20,554 |
| `lstm_state` | Unrestricted LSTM, previous predicted `Tin` fed back, state-dependent diffusion | 32 | 31,457 |

### Structure and uncertainty

For the constrained diagonal models,

\[
\tau_i=\exp\{\log(0.1)+\log(504/0.1)\operatorname{sigmoid}(a_i)\},
\qquad \rho_i=\exp(-0.25/\tau_i)<1.
\]

This permits fast and slow thermal modes. With fixed `m`, `xi`, prescribed inputs
and **common innovations**, the exogenous-noise model satisfies
\(\Delta h_{t+1}=\operatorname{diag}(\rho)\Delta h_t\): it is contractive by
construction. The state-dependent-noise model retains the pole bound, but the
diffusion difference prevents claiming that same pathwise contraction guarantee.

The unconstrained arm uses the bounded pole map below each initial raw pole and
its tangent continuation above it. Initial pole values and first derivatives
match the constrained arm. It introduces no extra parameters, pole clipping or
stability penalty; the upper bound is genuinely removed.

The structured noise scale is

\[
\sigma=10^{-5}+0.25\operatorname{sigmoid}
  \{r+0.1\,g(m,\xi,u_t,h_t^{\rm noise})\},
\]

where `h_noise=0` in both exogenous-noise arms and `h_noise=h_t` in the
state-dependent arm. The trainable base `r` participates in the active scale.
The paired structured models share the noise-network shape and initialization.
The LSTM applies state-conditioned innovations to its recurrent hidden output,
followed by `tanh` and an affine temperature readout. Its cell state and recurrent
mapping have no imposed contraction certificate.

All four models have two uncertainty sources: the fixed 4-D annual latent and
per-step Gaussian innovations. Independent innovations do **not** imply
independent temperature errors; recurrence, the shared latent and state-dependent
scales generate temporal dependence. There is no learned metadata-dependent
Gaussian prior in this thermal-only comparison, no HP emission noise and no
ex-post recalibration.

### Training and evaluation

- Two excitation policies; identical folds and seed; 10-day windows (`960`
steps), stride `2024`, batch size `32`.
- TBPTT every `32` steps (8 hours). This detaches gradients, not the forward
state; annual evaluation carries both state and latent continuously.
- Adam, learning rate `3e-4`, gradient norm clipping at `1`.
- `24` training particles, `16` validation particles and `100` annual particles.
- Shared objective: **fair RMS trajectory energy score + variogram score**
(order 0.5, lags 1/2/4/8, weight 1). No best-of-K term, noise-shrinkage penalty,
flexibility loss or sampled contraction penalty.
- Minimum 15 epochs, maximum 30; patience 5 with minimum improvement `1e-4`.
Select the minimum excitation-validation objective. Test/tariff scores never
select a checkpoint.
- Complete model and Adam states are saved for resumption. Numerical training
failures and nonfinite annual particles are recorded explicitly; failed
building ensembles are not silently filtered from aggregate rankings.

## 3b. Thermal-only results and conclusions

Results: [dashboard](../output/neural_building_emulator_refactor/ticino_v9_thermal_four_models_1000_e30/index.html), [trace scores](../output/neural_building_emulator_refactor/ticino_v9_thermal_four_models_1000_e30/summary.csv),
[annual KPIs](../output/neural_building_emulator_refactor/ticino_v9_thermal_four_models_1000_e30/annual_kpi_summary.csv), [publication figure](../output/neural_building_emulator_refactor/ticino_v9_thermal_four_models_1000_e30/publication_review/thermal_results_review.pdf) and
[paired building-bootstrap comparisons](../output/neural_building_emulator_refactor/ticino_v9_thermal_four_models_1000_e30/publication_review/paired_trace_effects.csv).

| Model | Epochs completed | Selected epoch | Annual evaluation |
| --- | ---: | ---: | --- |
| Constrained diagonal / exogenous noise | 30 | 29 | All 100 buildings finite under all four policies |
| Unconstrained diagonal / exogenous noise | 23 | 18 | Five buildings have nonfinite particles under every policy |
| Constrained diagonal / state-dependent noise | 30 | 30 | All finite |
| LSTM / state-dependent noise | 30 | 29 | All finite |

The unconstrained model stopped after five epochs without validation improvement,
not before the 15-epoch minimum. It had no skipped training updates. Nevertheless,
40/10,000 building–latent draws had a pole above one; 21 particles became
nonfinite per policy, affecting the same five buildings. Other finite trajectories
were also unusable, reaching approximately −5.8 × 10¹³°C. Its aggregate scores
are unavailable by design. This isolates an annual-stability benefit of the
constraint; it does not establish that every unconstrained architecture fails.

### Native trace accuracy and calibration

Cells below are **RMSE / CRPS / nominal 90% coverage**. RMSE and CRPS are in °C,
computed per building over the full year and then averaged across buildings.

| Policy | Structured / exogenous | Structured / state-dependent | LSTM |
| --- | --- | --- | --- |
| Excitation setpoint | 0.550 / 0.316 / 87.8% | 0.553 / 0.314 / 90.0% | 0.498 / 0.302 / 80.0% |
| Excitation force-off | 0.456 / 0.256 / 92.9% | 0.471 / 0.264 / 94.0% | 0.343 / 0.153 / 87.8% |
| Tariff setpoint | 0.497 / 0.288 / 90.1% | 0.485 / 0.276 / 93.3% | 0.488 / 0.268 / 83.3% |
| Tariff force-off | 0.446 / 0.251 / 93.2% | 0.459 / 0.258 / 94.4% | 0.327 / 0.145 / 88.2% |

The LSTM reduces force-off RMSE by 25–27% and CRPS by 40–42% relative to the
exogenous-noise structured model. Paired building-bootstrap intervals support
these gains. Its one-hour increment CRPS improves across all policies, and its
24-hour increment CRPS improves approximately 32–34% under force-off.

Setpoint results require the heating-availability breakdown. During those hours,
the state-dependent structured model has CRPS **0.324 versus 0.337** for LSTM on
excitation setpoint, and **0.263 versus 0.283** on tariff setpoint. LSTM's full-year
setpoint intervals undercover. Under force-off, LSTM heating-available coverage
is much closer to nominal: **89.4–90.1%**.

Adding state-dependent noise does not consistently improve trace accuracy over
exogenous noise. Its intervals are 12–14% wider; some undercoverage improves,
while other policies become conservative. The earlier 200-building ranking does
not persist uniformly in this larger comparison, which also changes training
budget and matched pole initialization.

### Annual temperature quantities

The scored annual quantities are mean `Tin` and

\[
D=0.25\sum_{t:\,\mathrm{SH\ available}}\max(T_{\mathrm{set},t}-T_t,0).
\]

`D` is a **zero-tolerance setpoint deficit**, not a regulatory comfort metric or
an electrical flexibility KPI. Cells below are **annual-mean-temperature CRPS /
its 90% coverage / deficit 90% coverage**.

| Policy | Structured / exogenous | Structured / state-dependent | LSTM |
| --- | --- | --- | --- |
| Excitation setpoint | 0.134 / 92% / 12% | 0.110 / 100% / 14% | 0.105 / 92% / 0% |
| Excitation force-off | 0.119 / 95% / 73% | 0.118 / 100% / 94% | 0.062 / 96% / 70% |
| Tariff setpoint | 0.144 / 92% / 19% | 0.108 / 100% / 55% | 0.122 / 91% / 6% |
| Tariff force-off | 0.120 / 95% / 25% | 0.115 / 100% / 48% | 0.061 / 96% / 20% |

Annual mean temperature can be well represented while accumulated deficits are
badly biased. The [deficit decomposition](../output/neural_building_emulator_refactor/ticino_v9_thermal_four_models_1000_e30/publication_review/deficit_decomposition_summary.csv) gives, for the
state-dependent structured model on setpoint excitation:

| Averaged across test buildings | Deficit, °C·h |
| --- | ---: |
| EnergyPlus | 76 |
| Deficit of the ensemble-mean trace | 1,064 |
| Mean deficit across sampled traces | 1,595 |

The heating-period mean temperature bias is only +0.003°C. Signed errors cancel
in an annual mean but negative excursions accumulate in `D`; ensemble spread
adds another 531°C·h through the convex threshold. Wider bands alone cannot
correct this mean-path error.

**Conclusion:** the bounded-pole construction protects long rollouts; LSTM is a
stronger force-off predictor; neither pointwise calibration nor trace CRPS
ensures calibration of annual threshold quantities. This experiment does not
prove a general superiority of physical regularization or establish performance
of autonomous `Tin,Qroom,Pel_SH` prediction.

## 4. The four full T,Q,P models and the two tank states

The current full comparison is [the fixed 15-epoch benchmark](../output/neural_building_emulator_refactor/ticino_v9_full_benchmark_1000_fixed_e15).
`T=Tin`, `Q=Qroom`, and `P=Pel_SH`. The two tank temperatures are auxiliary
outputs of **one** of the four neural models; they are not four extra models or
measured recurrent inputs shared by the competitors.

| Run variant | Registry model | Main architectural difference |
| --- | --- | --- |
| `baseline` | `closed_loop_hp_contracting_probabilistic` | Structured thermal dynamics, asymmetric Markov HP emission and a global-capacity buffer; historical “0.5-hour” recipe |
| `two_tanks` | Same registry model, `prob_tank_temperature_state=True` | Replaces the scalar buffer recurrence with effective SH and DHW temperature states and auxiliary supervision |
| `continuous` | `closed_loop_hp_vanilla_probabilistic_lstm` | Generic stochastic autoregressive LSTM with three continuous outputs |
| `hurdle_markov` | `closed_loop_hp_hurdle_markov_probabilistic_lstm` | Same type of recurrent core with continuous T/Q outputs and a Markov zero/positive HP-power emission |

### Structured buffer baseline

The thermal core is an eight-state, metadata/latent-conditioned diagonal model
with time constants in 0.1–504 hours and a learned forcing encoder. In contrast
to the isolated thermal-only experiment, its forcing and controller receive
generated temperature/heat feedback. Thermal process-noise scale depends on
state. The complete HP–buffer–thermal loop therefore **does not have a global
contraction certificate**, despite the historical registry name.

A 4-D base draw is transformed by a learned metadata-conditioned diagonal prior:

\[
z\sim\mathcal N(0,I_4),\qquad
\xi=\mu(m)+s(m)\odot z,\quad
\mu=\tanh(a(m)),\quad s=2\operatorname{sigmoid}(c(m)).
\]

It is held fixed through the annual rollout and shared across the active initial,
thermal, Q-to-T and HP pathways. These runs use shared routing, not the optional
separately routed factors, and have no colored-latent timescales enabled.

The HP model predicts asymmetric start/stop hazards. With previous mode `s_t`,

\[
\Pr(s_{t+1}=1)=a_t\{(1-s_t)p_{\rm start,t}+s_t(1-p_{\rm stop,t})\},
\qquad P_{t+1}=s_{t+1}P^{\rm active}_{t+1},
\]

where `a_t` encodes applicable generation availability. Active power has a
bounded positive distribution. Known force-off suppresses generation; room
heat can continue from storage. A learned monotone thermostat-demand component,
COP and energy-availability limit determine delivered heat.

The scalar buffer charges from `COP × Pel_SH`, loses heat, supplies the room and
is capped. **“0.5-hour” is a recipe label, not a building-specific measured
storage duration.** In this selected run `hp_buffer_capacity_mode=global` and the
resolved cap is **270.266 Wh/m²**, common across buildings. A positive explicit
`hp_energy_cap_wh_m2` takes precedence over the hours setting; it is not the
volume-conditioned tank model.

Uncertainty comprises the annual latent, thermal process innovations, Bernoulli
HP mode draws and active-power innovations. The optional `persistent_markov`
controller latent/noise is not active: the selected mode is `asymmetric_markov`.

Source: [probabilistic_contracting_closed_loop_hp.py](../neural_building_emulator/models/probabilistic_contracting_closed_loop_hp.py).

### Two effective tank temperatures

The two-tank arm retains the same thermal/HP framework and learns
\(\theta_{\rm SH},\theta_{\rm DHW}\). Effective heat capacities are anchored to
metadata volumes:

\[
C_{j,b}=1163\,\frac{V_{j,b}}{A_b}\,\kappa_j(m_b,\xi)
\quad[\mathrm{Wh/(m^2 K)}],\qquad 0.25\le\kappa_j\le4.
\]

Predicted SH temperature **conditions Qroom delivery** through a monotone
temperature-difference gate and available energy. Both predicted temperatures
also enter HP controller features. Thus these are active dynamical states, not
auxiliary labels disconnected from the T/Q/P predictions.

With effective net heat `q_net,j`, the generic store update is

\[
\theta_{j,t+1}=\operatorname{clip}_{[1,90]}
\left[T_a+(\theta_{j,t}-T_a)e^{-\lambda_j\Delta t}
+\frac{q_{{\rm net},j,t}}{C_j\lambda_j}(1-e^{-\lambda_j\Delta t})
+\sigma_{\rm tank}\epsilon_{j,t}\right].
\]

SH net heat combines predicted generation and delivered room heat with a learned
zone/building conversion. DHW net heat combines known withdrawal with recovery
anchored to `DHW_recovery_W`. Recovery is gated by generation availability,
predicted DHW temperature and the SH-active fraction. This is a generic
shared-source approximation, not an implementation of E+ stratified nodes or
exact mode scheduling.

Initial tank temperatures are learned from metadata/latent and seasonal
availability. Two independent per-step tank innovations share one learned noise
scale. The auxiliary observation is the midpoint of consecutive states, approximating the exported
interval-mean temperatures. Auxiliary marginal CRPS uses a fixed 10°C scale,
weight 1 and a four-hour initial exclusion within each training window.
The original three-target validation metric selects the checkpoint.

All four full models receive DHW volume/recovery metadata; only this arm has
tank states and tank supervision. The volume-conditioned capacities replace
the legacy buffer recurrence even though inherited configuration fields still
record the old scalar cap.

Sources: [tank_temperatures.py](../neural_building_emulator/models/tank_temperatures.py)
and [TANK_TEMPERATURE_EXPERIMENT.md](TANK_TEMPERATURE_EXPERIMENT.md).

### Vanilla and Markov-emission LSTMs

Both have 64 hidden units, metadata-conditioned initialization, a 4-D standard
Gaussian annual latent, and learned state-dependent process noise. At each step
they consume prescribed inputs, metadata, latent and their previous predicted
T/Q/P. Innovations perturb the recurrent hidden output before `tanh` and readout.
They have no explicit thermal storage or COP equations.

`continuous` predicts all three normalized outputs with a generic affine head.
It has no explicit zero/positive HP decomposition, hard physical positivity
constraint or external force-off generation gate; the requested command is an
input to be learned.

`hurdle_markov` keeps continuous T/Q heads but predicts separate start/stop
probabilities and bounded positive active power. Its Markov mechanism matches
the structured model's switching family, not an IID Bernoulli alternative.
The emission has its own parameterization and cap derived from training target
scales; it is not identical to the structured HP block. Its availability logic
includes seasonal availability and requested force-off.

Both LSTMs use the same ventilation-rule rollout as the structured models.
The `structured_gated` label in generic force-off report files is a historical
report key and does not turn an LSTM into a structured model.

Source: [models/lstm.py](models/lstm.py).

### Shared training, intentional differences and LightGBM

All four neural models use seed 13, the same v9 folds and two excitation policies,
10-day windows, batch size 32, Adam at `3e-4`, gradient clipping at 1, TBPTT at
32 steps, 24 training particles, 16 validation particles and 100 annual scoring
particles. Four seasonal windows from one policy trajectory share particle
latent identities for the pooled flexibility loss. Annual scoring has no window
resets. TBPTT is active for all four models; the probabilistic LSTM settings are
the relevant manifest section, not the separate deterministic-LSTM defaults.

The shared objective contains trajectory energy score, variogram score
(weight 1, order 0.5, lags 1/2/4/8, T/Q channels and constant-setpoint stretches),
soft-optimistic trajectory loss (weight 2), and direct-F CRPS (weight 1.5).
The direct-F training term is setpoint-event based and pools four seasonal
windows; force-off KPI evaluation is separate. Increment/event energy-score
options are disabled in this recipe.

Structured models and Markov LSTM also use HP mode loss (0.1), active-power NLL
(1) and inactive-leakage penalty (0.1), with `hard_expected_grad` switching during
training and sampled Bernoulli switching at inference. Vanilla LSTM has no HP
head, so those losses are zero. Two-tank CRPS is another deliberate difference.

The legacy structured `prob_physics_weight=0.1` controls a process-scale penalty,
not a general physical residual loss. Its effective parameter dependence differs
from the LSTM penalty; see open points. The sampled local contraction penalty
and monotonicity-loss weights are **zero**. Structural restrictions, auxiliary
supervision, gradient handling and explicit loss regularization must not be
conflated in publication claims.

All neural models completed **15 epochs**, with selected checkpoints:

| Buffer baseline | Two tanks | Vanilla LSTM | Markov LSTM |
| ---: | ---: | ---: | ---: |
| 13 | 13 | 15 | 14 |

Selection minimizes three-target validation NRMSE, not test CRPS. The structured
models were replayed with verified earlier checkpoints to recover live Adam
history and complete the 15-epoch budget; the already completed 15-epoch LSTM
fits and LightGBM models were reused. The frozen selected artifacts, rather than
inherited top-level recipe defaults, identify the actual trained configuration.

**LightGBM is an additional KPI-only benchmark.** Separate metadata-only
quantile regressors predict each policy-family/KPI/direction/horizon cell.
Quantiles 0.16/0.50/0.84 provide a Gaussian location-scale sample approximation.
It generates no T/Q/P traces and has no learned trajectory joint distribution.
Source-policy models are applied unchanged to the corresponding tariff policy;
they do not receive its command trajectory or policy summary features.

Runner/protocol sources: [fixed_epoch_campaign.py](fixed_epoch_campaign.py),
[campaign_benchmark.py](campaign_benchmark.py),
[probabilistic_lstm_trainer.py](probabilistic_lstm_trainer.py), and
[CALIBRATED_MODEL_BENCHMARK.md](CALIBRATED_MODEL_BENCHMARK.md).

## 4b. Full-emulator results and conclusions

Results: [visual report](../output/neural_building_emulator_refactor/ticino_v9_full_benchmark_1000_fixed_e15/visual_summary/index.html), [trace scores](../output/neural_building_emulator_refactor/ticino_v9_full_benchmark_1000_fixed_e15/visual_summary/trace_summary.csv),
[KPI scores](../output/neural_building_emulator_refactor/ticino_v9_full_benchmark_1000_fixed_e15/visual_summary/kpi_summary.csv), [joint scores](../output/neural_building_emulator_refactor/ticino_v9_full_benchmark_1000_fixed_e15/visual_summary/joint_scores.csv) and
[paired comparisons against LightGBM](../output/neural_building_emulator_refactor/ticino_v9_full_benchmark_1000_fixed_e15/visual_summary/publication_review/paired_crps_vs_lightgbm.csv). These are the completed
1,000-building, fixed-15-epoch results, not the earlier 200-building screen.

The [annual trajectory explorer](../output/neural_building_emulator_refactor/ticino_v9_full_benchmark_1000_fixed_e15/annual_trace_explorer/index.html)
collects five test buildings across all four models and policies. It provides
linked T/Q/P panels, native 95% predictive bands and requested force-off shading
on the power panel. Open the HTML locally; keep its supporting files alongside
it. [annual_trace_explorer.py](annual_trace_explorer.py) rebuilds the viewer from
saved annual plots without rerunning inference.

### Native T/Q/P traces

Every triple is ordered **T / Q / P**. Units are °C for T and W/m² for Q/P.
RMSE is the mean building RMSE over all annual timestamps. CRPS and rank-based
central-interval coverage use all timestamps for T but **space-heating-available
timestamps for Q/P**, following the saved report's masks. These different
aggregation scopes should be retained when quoting results.

| Policy | Model | RMSE T / Q / P | CRPS T / Q / P | 90% coverage T / Q / P |
| --- | --- | --- | --- | --- |
| Excitation setpoint | Buffer baseline | 0.372 / 2.330 / 7.336 | 0.179 / 1.660 / 3.502 | 77.3% / 56.9% / 89.9% |
| Excitation setpoint | Two tanks | 0.402 / 2.425 / 7.317 | 0.198 / 1.766 / 3.482 | 75.1% / 73.9% / 89.7% |
| Excitation setpoint | Vanilla LSTM | 0.375 / 2.275 / 7.220 | 0.182 / 1.497 / 4.145 | 91.1% / 93.8% / 94.1% |
| Excitation setpoint | Markov LSTM | 0.384 / 2.246 / 7.192 | 0.180 / 1.525 / 3.410 | 80.5% / 54.6% / 89.2% |
| Excitation force-off | Buffer baseline | 0.509 / 3.146 / 6.186 | 0.229 / 2.063 / 2.579 | 73.2% / 52.5% / 90.1% |
| Excitation force-off | Two tanks | 0.533 / 3.538 / 6.065 | 0.232 / 2.122 / 2.493 | 79.1% / 76.4% / 90.6% |
| Excitation force-off | Vanilla LSTM | 0.465 / 2.550 / 5.612 | 0.203 / 1.606 / 2.840 | 89.6% / 92.6% / 94.3% |
| Excitation force-off | Markov LSTM | 0.475 / 2.503 / 5.631 | 0.211 / 1.671 / 2.199 | 76.4% / 50.4% / 89.9% |
| Tariff setpoint | Buffer baseline | 0.370 / 2.479 / 7.065 | 0.183 / 1.720 / 3.304 | 75.4% / 55.4% / 90.1% |
| Tariff setpoint | Two tanks | 0.406 / 2.555 / 7.043 | 0.204 / 1.787 / 3.282 | 74.7% / 72.8% / 90.4% |
| Tariff setpoint | Vanilla LSTM | 0.377 / 2.669 / 6.880 | 0.185 / 1.628 / 3.887 | 91.2% / 93.1% / 94.8% |
| Tariff setpoint | Markov LSTM | 0.384 / 2.656 / 6.827 | 0.186 / 1.743 / 3.144 | 78.3% / 52.2% / 89.7% |
| Tariff force-off | Buffer baseline | 0.380 / 2.301 / 6.831 | 0.179 / 1.661 / 3.064 | 78.2% / 55.5% / 90.4% |
| Tariff force-off | Two tanks | 0.396 / 2.432 / 6.770 | 0.188 / 1.730 / 3.014 | 79.2% / 76.3% / 90.2% |
| Tariff force-off | Vanilla LSTM | 0.377 / 1.967 / 6.349 | 0.176 / 1.373 / 3.400 | 91.9% / 96.0% / 94.5% |
| Tariff force-off | Markov LSTM | 0.385 / 1.868 / 6.361 | 0.177 / 1.353 / 2.716 | 80.3% / 55.1% / 89.8% |

The buffer baseline remains competitive for setpoint temperature accuracy.
Vanilla LSTM gives much better native T/Q coverage, although Q and P intervals
can be conservative. Markov LSTM gives the best P CRPS in every policy and
strong force-off Q accuracy, but its Q intervals severely undercover.

Two tanks improve Q coverage substantially: approximately 73–76%, versus 53–57%
for the buffer baseline. That is still below 90%, and it is **not an aggregate
T/Q accuracy win**: T and Q RMSE are worse than the baseline under all four
policies. Better-looking individual force-off drops therefore cannot establish
overall superiority. The latest shared 1,000-building report does not contain
an aggregate SH/DHW-temperature calibration table; those auxiliary outputs need
their own audit before claiming accurate tank-state reconstruction.

Temporal diagnostics also remain imperfect. In the setpoint-excitation HP
diagnostic, the true mean on-run duration is 0.657 h: baseline predicts 0.461 h,
two tanks 0.435 h, Markov LSTM 0.448 h and vanilla LSTM 1.152 h. Good marginal
power coverage does not ensure correct switching durations or autocorrelation.
Those detailed HP diagnostics are present for setpoint-family policies; missing
force-off-family summary entries should not be interpreted as zeros.

### Native flexibility KPIs

Both definitions are evaluated at 0.5, 1, 2 and 3 hours. Setpoint KPIs have up/down
directions; force-off KPIs measure shed power. Direct F is the horizon times a
ratio of aggregated power response to intervention dose. The regression version
adjusts for the configured event/context covariates. These are policy-dependent
response estimators, not necessarily invariant intrinsic building constants.

Cells below are **CRPS / nominal 90% coverage**, equally averaged over the
direction/horizon cells within each family (eight setpoint cells or four force-off
cells, each with 100 test buildings). Setpoint units are Wh/(m²·K); force-off
units are Wh/(m²·intervention). Do not pool their absolute scores into one number.

| Policy | KPI | Buffer baseline | Two tanks | Vanilla LSTM | Markov LSTM | LightGBM |
| --- | --- | --- | --- | --- | --- | --- |
| Excitation setpoint | Regression | 0.331 / 75.6% | 0.222 / 91.1% | 0.258 / 85.5% | 0.282 / 83.0% | 0.276 / 81.6% |
| Excitation setpoint | Direct F | 0.323 / 72.5% | 0.272 / 85.0% | 0.284 / 79.6% | 0.261 / 80.0% | 0.269 / 83.2% |
| Excitation force-off | Regression | 0.571 / 83.8% | 0.513 / 66.0% | 0.515 / 94.2% | 0.304 / 74.2% | 0.473 / 83.0% |
| Excitation force-off | Direct F | 0.508 / 88.8% | 0.540 / 85.0% | 0.514 / 93.2% | 0.301 / 81.5% | 0.478 / 80.0% |
| Tariff setpoint | Regression | 0.638 / 72.2% | 0.617 / 73.6% | 0.676 / 72.8% | 0.722 / 66.3% | 1.023 / 28.1% |
| Tariff setpoint | Direct F | 0.596 / 47.5% | 0.673 / 60.8% | 0.463 / 76.8% | 0.354 / 66.1% | 1.683 / 28.5% |
| Tariff force-off | Regression | 0.592 / 87.2% | 0.499 / 77.2% | 0.522 / 91.5% | 0.425 / 74.2% | 0.843 / 58.5% |
| Tariff force-off | Direct F | 0.474 / 87.5% | 0.412 / 89.0% | 0.486 / 88.5% | 0.263 / 85.5% | 1.959 / 26.0% |

The two-tank model is strongest for native setpoint-excitation regression CRPS
(0.222), improving on baseline 0.331 and LightGBM 0.276. It does not win every
flexibility definition: its force-off-excitation direct-F CRPS is 0.540, versus
0.508 for baseline and 0.301 for Markov LSTM.

Markov LSTM is the strongest native direct-F predictor across all four policies
and wins both force-off KPI families. The two-tank model gives the best native tariff-
setpoint regression CRPS among the neural models. Thus the preferred model
depends on the output and flexibility definition; no neural model dominates all
accuracy and calibration criteria.

The tariff advantage over the **frozen metadata-only LightGBM** is large.
Markov LSTM reduces native direct-F CRPS by **79.0%** on tariff setpoint and
**86.6%** on tariff force-off; paired building-bootstrap 95% intervals are
76.8–81.0% and 84.3–88.5%. Regression reductions are 29.4% and 49.5%.
This supports the value of trajectory-aware emulation under changed policies.
It does **not** isolate superiority over a tree model trained with comparable
policy information and policy diversity: this LightGBM baseline cannot change
its prediction for the same metadata when the policy changes.

### RNS and RNS–ECC: separate KPI post-processing

For each model/KPI cell, RNS fits signed standardized residuals on the 99 source-
policy calibration buildings, using ensemble median and `IQR/1.349` with a scale
floor. The same fitted calibrator is transferred to the tariff policy. It changes
KPI locations/spreads, not annual trajectories. RNS–ECC assigns those calibrated
marginal quantiles back to raw particle labels by rank:

\[
\widetilde K_{d,k}=q_{d,\operatorname{rank}_k(K_{d,k})}.
\]

Consequently **RNS and RNS–ECC have identical marginal CRPS, coverage and
quantiles**. ECC preserves the raw cross-KPI rank structure and is assessed with
joint scores. It does not make the adjusted KPIs belong to actual generated
physical trajectories, and it is not whole-trajectory reweighting.

For example, the Markov LSTM's joint energy scores are:

| Policy | Native | RNS | RNS–ECC |
| --- | --- | --- | --- |
| Excitation setpoint | 1.388 | 1.109 | 1.045 |
| Excitation force-off | 0.293 | 0.260 | 0.255 |
| Tariff setpoint | 2.887 | 2.320 | 2.156 |
| Tariff force-off | 0.357 | 0.335 | 0.321 |

Compare these only within a policy: setpoint vectors have 16 coordinates and
force-off vectors have eight, with source-fitted standardization. The joint
score improvement from ECC is additional to RNS, not another marginal gain.

Source-policy recalibration often helps: Markov LSTM force-off-excitation
regression coverage rises from 74.25% to 91.75%, while CRPS falls from 0.304 to
0.260. Transfer is less reliable. Its tariff-force-off direct-F CRPS worsens
from 0.263 raw to 0.284 after RNS; for two tanks the same score worsens from
0.412 to 0.653. Calibrated tariff-setpoint direct-F coverage for Markov LSTM is
still only 76.4%. Calibration on excitation policies does not guarantee
calibration under a new policy.

**Conclusion:** adding an appropriate HP emission to an LSTM is valuable for
electrical power and flexibility; adding explicit tank states improves some
coverage and regression KPIs but is not an overall accuracy win. Native traces,
native KPI distributions, post-calibrated KPIs and physical trajectory coherence
are separate results and should remain separate in the paper.

## 5. Open points

1. **Annual functionals and persistent uncertainty.** Diagnose threshold crossing,

   cumulative energy and response-location errors as well as interval width.
    Repeat the latent-versus-fast-noise variance decomposition on current
    checkpoints, per building with deterministic inputs fixed, including absolute
    and normalized variance, PIT/coverage, latent scales and operating regimes.
    Low latent effective rank alone is not evidence that more latent dimensions
    are needed; compare it with the standardized held-out residual covariance.
2. **Annual validation and rare instability.** Short-window validation selected an

   unconstrained checkpoint with catastrophic annual tails. Add representative
    continuous annual validation and explicit pole/state diagnostics before
    promotion. Retain failures and implausible finite trajectories in the audit;
    a finite-value check or central confidence band alone is insufficient.
3. **Optimization and strength of architectural claims.** Both comparisons use

   one training seed. In the thermal run, structured arms clip gradients in about
    97% of batches versus 2.6% for LSTM, and LSTM has 31,457 versus 20,554 parameters.
    Repeat seeds and examine clipping/learning-rate and capacity sensitivity with
    comparable validation tuning budgets. A shared epoch ceiling is not proof of
    convergence. Separate simulation stability from training stabilization;
    TBPTT, clipping and the full model's 0.05 temperature-feedback gradient scale
    do not certify forward-dynamics contraction.
4. **Legacy process-noise regularization mismatch.** In the full structured

   heteroscedastic model, the active scale uses `process_noise_init + 0.1*net(...)`,
    while the nominal process penalty acts on `raw_process_scale`, which is unused
    by that branch. LSTMs use that trainable base in the active scale. A saved
    [sensitivity audit](../output/neural_building_emulator_refactor/ticino_v9_full_benchmark_1000_fixed_e15/visual_summary/publication_review/process_regularization_sensitivity.json) confirms the difference. Correct and ablate
    this in a new controlled run; do not relabel the existing results as perfectly
    matched regularization. The new thermal-only experiment avoids this penalty
    and uses an active trainable base scale in all arms.
5. **Tank-state identifiability and delivery behavior.** Score SH/DHW means,

   coverage, clipping frequency, depletion and recovery on the selected full-run
    checkpoint. Separate the effect of two physical state variables from the
    effect of two extra supervision targets. Validate the generic shared-source
    and zone/building conversion assumptions without reproducing E+ internals.
6. **Loss identification.** The full benchmark contains a setpoint direct-F term,

   but no matching force-off KPI training term, and contains a soft-optimistic
    component that is not by itself a proper probabilistic score. Test targeted
    annual/threshold and force-off objectives separately from routing changes.
    Do not infer that broader noise will remove deterministic response bias.
7. **Fair policy-aware baselines and estimator validity.** A policy-aware LightGBM

   comparison needs identifiable policy features and enough training-policy
    variation. Adding nearly constant summaries to one excitation regime may be
    insufficient. Preserve matched buildings/weather, inspect event counts and
    covariate overlap, and distinguish prediction of the policy-specific estimator
    from an invariant causal flexibility property. Current transfer gains establish
    an advantage over this metadata-only baseline, not every possible tree model.
8. **Calibration versus trajectory coherence.** Test source-only recalibration

   under shift and joint KPI calibration without fitting on tariff test outcomes.
    Whole-trajectory weighting is a separate candidate requiring adequate native
    support and effective-sample-size diagnostics; it is not what RNS–ECC does.
    None of the results here establishes calibrated physical traces from
    post-calibrated KPI samples.
9. **Publication uncertainty and reporting scope.** Bootstrap buildings, retaining

   their repeated policies and KPI cells together. Existing intervals are
    conditional on fixed trained checkpoints and sampled cities/weather, not
    uncertainty over training seeds or new municipalities. Report heating-available
    and full-year scopes, native versus post-calibrated results, failure incidence,
    and both direct-F and regression definitions. Favor effect sizes and
    prespecified comparisons over declaring a winner from many exploratory tests.

### Provenance and navigation

The thermal runner stores source snapshots, hashes, the split, train-only scalers,
selected weights, optimizer-resume states and per-building stability diagnostics.
The full runner stores selected artifact manifests, replay verifications, source
hashes and separate source/transfer reports. Historical absolute paths in inherited
recipes are not authoritative over the selected artifact's resolved data/config.
The links below require the local data/output directories; those artifacts may
not be included in a source-only checkout.

The report and dashboard generators are versioned alongside the models. From the
repository root, regenerate the saved-score report or start the live training board
with:

```sh
.venv/bin/python -m neural_building_emulator_refactor.benchmark_visual_report \
  --run-dir output/neural_building_emulator_refactor/ticino_v9_full_benchmark_1000_fixed_e15
.venv/bin/python -m neural_building_emulator_refactor.live_training_board \
  --run output/neural_building_emulator_refactor/ticino_v9_full_benchmark_1000_fixed_e15
.venv/bin/python -m neural_building_emulator_refactor.thermal_publication_review \
  --run-dir output/neural_building_emulator_refactor/ticino_v9_thermal_four_models_1000_e30
```

The first and third commands summarize existing scores, without model fitting.
They retain the published protocol's checks for 100 matched test buildings. The
live board writes an offline HTML page and refreshes its local snapshot while the
campaign runs; `--once` writes a single snapshot. These tools require the local
run artifacts and the repository's scientific Python environment. The annual
trajectory explorer has its own CLI in
[annual_trace_explorer.py](annual_trace_explorer.py); its HTML and JavaScript,
and the report/monitor templates, live in [viewer_assets](viewer_assets/).

Older implementation details remain in
[README_MODELS.md](../neural_building_emulator/README_MODELS.md),
[CALIBRATED_MODEL_BENCHMARK.md](CALIBRATED_MODEL_BENCHMARK.md) and
[TANK_TEMPERATURE_EXPERIMENT.md](TANK_TEMPERATURE_EXPERIMENT.md). Their historical
defaults should not override the executed configurations summarized here.
