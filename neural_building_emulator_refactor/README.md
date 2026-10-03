# Neural Building Emulator Refactor

This package is a parallel comparison pipeline. It does not replace or modify
`neural_building_emulator`; the established state-space equations and trainer
remain available there. The refactor adds a small registry, common artifacts,
recurrent baselines, and one reload-and-score interface.

## Models

| Registry name | Task | Implementation |
|---|---|---|
| `q_to_t_deterministic_ss` | known `Qroom`, weather -> `Tin` | existing deterministic stable state-space model |
| `q_to_t_probabilistic_ss` | known `Qroom`, weather -> `Tin` | existing probabilistic stable state-space model |
| `closed_loop_hp_deterministic_ss` | `Tset`, weather -> `Tin,Qroom,Pel` | existing deterministic HP/buffer plus stable thermal model |
| `closed_loop_hp_probabilistic_ss` | `Tset`, weather -> `Tin,Qroom,Pel` | existing probabilistic HP/buffer plus stable thermal model |
| `closed_loop_hp_contracting_deterministic` | `Tset`, weather -> `Tin,Qroom,Pel` | existing bounded/contracting deterministic model |
| `closed_loop_hp_contracting_probabilistic` | `Tset`, weather -> `Tin,Qroom,Pel` | existing bounded/contracting closed-loop model |
| `q_to_t_lstm` | known `Qroom`, weather -> `Tin` | new unconstrained autoregressive LSTM |
| `closed_loop_hp_lstm` | `Tset`, weather -> `Tin,Qroom,Pel` | new unconstrained autoregressive LSTM |
| `closed_loop_hp_vanilla_probabilistic_lstm` | `Tset`, weather -> distributions of `Tin,Qroom,Pel` | unstructured stochastic autoregressive LSTM comparison baseline |
| `closed_loop_hp_probabilistic_lstm` | `Tset`, weather -> distributions of `Tin,Qroom,Pel` | older semi-structured stochastic LSTM retained for reproducibility |

The two deterministic LSTMs are intentionally unstructured controls. Static metadata initializes
their hidden and cell states. During autonomous rollout, the previous predicted
output is fed back: `Tin` for Q-to-T and `[Tin,Qroom,Pel]` for closed loop. They
have no stability, positivity, energy-conservation, or monotonicity constraints.
The vanilla probabilistic LSTM uses the same autoregressive inputs and a generic
linear three-output readout. Its scenarios come from trajectory-persistent
latent noise and time-varying recurrent process noise. It has no engineered
setpoint-shock traces, availability gate, positivity transform, HP switching
state, active-power distribution, buffer, COP, thermal state-space,
contraction, or monotone thermostat equation. It is trained only with the
shared multivariate energy score, optional variogram score, and optional
trajectory Best-of-K term. This is the intended control for testing whether
the structured emulator is necessary.

The older `closed_loop_hp_probabilistic_lstm` is deliberately kept loadable so
existing results remain reproducible. It is not a vanilla baseline: it includes
positive bounded `Qroom`, setpoint-shock traces, and an asymmetric-Markov,
bounded-active-power model for `Pel`.

Contracting closed-loop state-space models also support the optional simplified
temperature readout:

```bash
--contracting-temperature-update bounded_equilibrium \
  --contracting-temperature-delta-max-c 2.0
```

It removes the learned outer temperature `alpha` and moves toward the decoded
equilibrium through a smooth hard step cap. Existing configurations continue
to use their saved update mode.

Use `alpha_bounded_equilibrium` instead to retain a metadata-conditioned
rollout-constant `alpha` inside the bounded update. In probabilistic models it
is additionally conditioned on persistent `xi`, and is therefore constant only
within each sampled trajectory.

Contracting models can damp only the backward derivative of recursive predicted
temperature feedback with:

```bash
--contracting-temperature-feedback-gradient-scale 0.05
```

The predicted `Tin` values supplied to the HP/Q controller and thermal
transition are unchanged in the forward pass and at inference. A value of `1`
preserves ordinary gradients, while `0` detaches those feedback paths. The
direct temperature recurrence remains differentiable. This is a training
stabilizer, not a forward-dynamics contraction certificate.

All models use the original profile-level split and preprocessing code.
Q-to-T is aligned as `u[t] -> Tin[t+1]`. For timestamped EnergyPlus interval
reports, closed loop is aligned as
`Tin_previous, u_interval -> [Tin_end, Qroom_interval, Pel_SH_interval]`.
Powers use W/m2, and closed-loop data
uses the same HP-only filter and DHW masking as the established trainer.
`Qroom` is normalized by the modeled zone's `floor_area`; whole-building
`Pel_SH` is normalized by `floor_area * totalFloors`. New runs use this corrected
normalization by default. Pass `--hp-power-area-normalization zone_floor_area`
only when reproducing a legacy run. Saved artifacts preserve the choice, and
old artifacts without this metadata field are treated as legacy artifacts.
The contracting probabilistic model can optionally add a separate hysteretic
hydronic-demand state with
`--prob-qroom-activation-model asymmetric_markov`; this also requires
`--hp-thermostat-demand-mode monotone`. Its start/stop loss is controlled by
`--qroom-mode-loss-weight` and the observed on threshold by
`--qroom-on-threshold-w-m2`. The default `continuous` value preserves the
existing architecture.
Closed-loop state-space models also receive HP capacity, SH design capacity and
SH buffer volume normalized by that heated area, together with reference COP
and a one-hot HP model identifier. Their default structural Q-to-T filter bank
uses fixed time constants `0.25, 1, 4, 16, 24, 48` hours.

The data layer accepts both the legacy consolidated wide parquet dataset and
the entity-oriented `egid=*/timeseries.parquet` simulation folders. In the new
format, the modeled temperature, setpoint, room heat, ventilation, internal
gain, and occupancy come from `fl0_thz0`; weather comes from `building_total`;
HP electricity and operating mode come from `technical_room`; static metadata
comes from `simulation_metadata.json`. Internal gains and occupancy enter every
model encoder as W/m2 and persons/m2 respectively. Legacy simulations without
these channels remain loadable and use zero-valued forcing channels.

## Train

List the registry:

```bash
.venv/bin/python -m neural_building_emulator_refactor
```

Train the unstructured closed-loop baseline:

```bash
.venv/bin/python -m neural_building_emulator_refactor.train \
  --model closed_loop_hp_lstm \
  --dataset neural_building_emulator/tessin_results.parquet/all_hp \
  --output-dir output/neural_building_emulator_refactor/closed_loop_lstm \
  --max-profiles 1000 \
  --test-fraction 0.1 \
  --epochs 20 \
  --batch-size 32 \
  --sequence-length 960 \
  --stride 2024 \
  --lstm-hidden-dim 64
```

Train a state-space model using a checked-in legacy configuration:

```bash
.venv/bin/python -m neural_building_emulator_refactor.train \
  --model closed_loop_hp_contracting_probabilistic \
  --dataset neural_building_emulator/tessin_results.parquet/all_hp \
  --output-dir output/neural_building_emulator_refactor/contracting \
  --max-profiles 1000 \
  --test-fraction 0.1 \
  --epochs 20 \
  --batch-size 32 \
  --sequence-length 960 \
  --stride 2024 \
  --legacy-config-json neural_building_emulator_refactor/configs/closed_loop_hp_contracting_probabilistic.json
```

The JSON may contain any `neural_building_emulator.train.TrainConfig` field.
The registry adapter always controls model kind, dataset, split, optimization,
windowing, output directory, and checkpoint saving from the concise CLI.

The exact legacy options recovered from the successful
`qroom_monotone_dynamic_matrix_positive_leaky_1000` artifact are preserved in
`configs/qroom_monotone_dynamic_matrix_positive_leaky_1000.json`. This is a
run-specific reproduction config; it intentionally differs from the generic
contracting example.

Replay that pre-refactor probabilistic recipe through the registry adapter with:

```bash
.venv/bin/python -m neural_building_emulator_refactor.train \
  --model closed_loop_hp_contracting_probabilistic \
  --dataset neural_building_emulator/tessin_results.parquet/all_hp \
  --output-dir output/neural_building_emulator_refactor/reproduce_old_positive_leaky_1000 \
  --max-profiles 1000 \
  --test-fraction 0.2 \
  --epochs 15 \
  --batch-size 32 \
  --learning-rate 3e-4 \
  --gradient-clip-norm 1.0 \
  --train-eval-max-windows 64 \
  --sequence-length 960 \
  --stride 2024 \
  --rotate-window-starts \
  --hp-power-area-normalization zone_floor_area \
  --seed 13 \
  --legacy-config-json neural_building_emulator_refactor/configs/qroom_monotone_dynamic_matrix_positive_leaky_1000.json
```

The JSON pins the observed training-set HP caps and the old effective checkpoint
criterion. The historical artifact recorded `checkpoint_metric=auto`, which at
that time resolved to `test_rmse_c`; the reproduction explicitly stores
`test_rmse_c` because current `auto` behavior prefers total normalized RMSE.
The explicit legacy area mode is also required because that run divided
whole-building HP electric power by the modeled zone area.

For a fresh retrain with corrected whole-building HP normalization, use
`configs/qroom_monotone_dynamic_matrix_positive_leaky_building_area_1000.json`
instead. It keeps `contracting_q_to_t_mode=positive_leaky` but sets the three
data-derived HP, room-heat, and energy caps to zero so the trainer recomputes
them under the corrected target scaling. Use a new `--output-dir`; the original
reproduction artifact is not overwritten.
Expected data parity is 770 train and 193 test profiles after the common HP
filter, with the same ordered profile IDs as the original seed-13 split.

Frequently tuned state-space options are also exposed directly with their old
flag names, including encoder/state dimensions, contracting and positive-leaky
settings, probabilistic particles/noise/HP emission, variogram and IRES terms,
TBPTT, gradient guards, evaluation scheduling, and plot counts. The original
`--model-kind` flag is accepted as an alias for the corresponding registry
entry. Explicit CLI values override values from `--legacy-config-json`.

The contracting models expose three transition-conditioning modes:
`state_feedback`, `exogenous`, and `exogenous_additive_feedback`. The last mode
generates bounded `M_t`, `B_t`, and `b_t` from exogenous conditions while
injecting encoded `Tin` and `Tout-Tin` only through `B_t z_t`. Its input-matrix
bound is configured with `--contracting-additive-feedback-gain-bound`.

## Artifacts

Every selected model is stored under:

```text
<output-dir>/artifacts/<model-name-or-legacy-kind>/selected/
```

Each selected directory is self-describing. `load_artifact(path)` reconstructs
both new LSTMs and state-space checkpoints. Existing state-space artifacts that
predate this package can also be loaded directly when their model kind maps
unambiguously to this registry.

Python use:

```python
from pathlib import Path
from neural_building_emulator_refactor.artifacts import load_artifact

artifact = load_artifact(Path("output/.../selected"))
print(artifact.spec.name, artifact.metadata)
```

## Score

Score full autonomous test profiles and compute flexibility KPIs for a
closed-loop probabilistic model:

```bash
.venv/bin/python -m neural_building_emulator_refactor.score \
  --artifact-dir output/neural_building_emulator_refactor/contracting/artifacts/closed_loop_hp_contracting_probabilistic/selected \
  --dataset neural_building_emulator/tessin_results.parquet/all_hp \
  --output-dir output/neural_building_emulator_refactor/contracting/scores \
  --profile-source test \
  --kpi-mode scenario_average \
  --num-scenarios 100
```

Outputs include `summary.json`, per-profile physical metrics, event-study KPI
tables, `flexibility_event_study_comparison.html`, per-scenario KPIs, and a KPI
quantile-coverage HTML plot for probabilistic artifacts. With
`--kpi-mode mean`, the KPI is fitted to the mean predicted trajectory. With
`scenario_average`, it is fitted separately to linked `Tin,Qroom,Pel` scenarios
and averaged afterward.

When regression-based flexibility is the primary estimator, add
`--also-evaluate-direct-f` to compute the direct-ratio annual
upward/downward `F_H` KPIs from the same trajectories. These secondary outputs
use the `_f` suffix, including `flexibility_kpis_f.csv` and
`flexibility_kpi_coverage_f.html`; no additional emulator rollout is required.

The scorer also writes common full-profile HTML plots (`Tin` for Q-to-T and
three stacked `Tin/Qroom/Pel` panels for closed loop). Probabilistic plots show
the scenario mean and pointwise 95% interval. Control their count with
`--num-profile-plots`.

Q-to-T models are scored on temperature but do not receive a flexibility KPI:
their heat input is observed, so they do not generate the electrical response
to a setpoint intervention.

For the contracting probabilistic closed-loop model, the same calibration can
be launched automatically after training by adding:

```bash
--evaluate-flexibility-after-training
```

This convenience preset evaluates up to 100 saved test profiles using 100
linked scenarios, Bernoulli HP sampling, the emission mode saved in the
artifact, the EnergyPlus ventilation rule, horizons `0.5,1,2,3` h, full
regression controls, a `0.05` C event threshold, at least 20 events, and seed
13. Its default estimator is `regression`, matching the effective final value
of the historical command that specified `--kpi-estimator` twice. Override the
profile cap, estimator, or destination with
`--post-training-flex-max-profiles`,
`--post-training-flex-kpi-estimator`, and
`--post-training-flex-output-dir`. The default destination is
`flexibility_event_study_scenarios_regressor/` under the training output.
The scenario-average evaluation also writes
`temporal_trace_calibration.html`, a three-signal reliability and ensemble-rank
dashboard for `Tin`, `Qroom`, and `Pel_SH`, plus the corresponding calibration
and rank-histogram CSV files. It also writes
`temporal_trace_probabilistic_scores.csv` with exact empirical ensemble CRPS
and mean 50/80/90/95% interval widths for each signal. Temperature uses all timestamps; the two heating
signals use only timestamps when space heating is available so the diagnostic
is not dominated by structurally zero off-season predictions.

The same standard post-training evaluation uses the shared benchmark scorer.
It additionally writes `pel_dynamics_calibration.html` with HP start/stay
reliability, conditional power calibration, active-run durations, mode
autocorrelation, and annual electrical-energy calibration. It also writes five
full-year profile plots and, when regression flexibility is selected, the
direct-ratio F comparison and coverage plots with the `_f` suffix.

### Metadata-only flexibility benchmark

To test whether the closed-loop emulator is needed when only the final
building-level flexibility index is required, fit LightGBM directly from the
same 15 static metadata fields to EnergyPlus event-study KPIs:

```bash
.venv/bin/python -m neural_building_emulator_refactor.metadata_flexibility_lightgbm \
  --artifact-dir output/.../artifacts/closed_loop_hp_contracting_probabilistic/selected \
  --output-dir output/neural_building_emulator_refactor/metadata_flexibility_lightgbm \
  --kpis f regression_based \
  --additional-emulator-artifact vanilla_lstm=output/.../artifacts/closed_loop_hp_vanilla_probabilistic_lstm/selected \
  --horizons-hours 0.5 1 2 3 \
  --controls full
```

The artifact supplies its training dataset path, exact building-level
train/test split, metadata columns, and power-normalization convention. Pass
`--dataset` only to intentionally override the saved dataset. KPI targets
are calculated from EnergyPlus only; LightGBM receives no time-series inputs,
weather, or emulator predictions. Separate models are fitted for upward and
downward flexibility at each horizon and for each requested KPI definition.
`f` selects the direct-ratio energy-flexibility KPI, while `regression_based`
selects the event-study coefficient. Their artifacts are written under `f/`
and `regression_based/`; combined metrics and a cross-KPI dashboard are written
at the output root. The older single-KPI `--kpi-estimator` interface remains
supported. Outputs include reloadable LightGBM text models, per-building
predictions, metrics, paired bootstrap comparisons with the emulator, feature
importance, and interactive held-out scatter/score dashboards.

When no `--emulator-kpi-csv` is supplied, the benchmark reloads the artifact,
runs its complete test profiles once, and reuses those trajectories for every
requested KPI. Probabilistic artifacts use 100 linked scenarios, Bernoulli HP
emissions, and recorded ventilation by default. These can be changed with
`--emulator-num-scenarios`, `--emulator-hp-scenario-mode`, and
`--emulator-ventilation-rollout-mode`. Generated scoring files are stored in
`emulator_scoring/`. Supplying explicit `f=PATH` and
`regression_based=PATH` CSVs skips this potentially expensive rollout.
Repeat `--additional-emulator-artifact NAME=PATH` to include other trained
closed-loop models. Each additional model is rolled out once and appears as a
separate method in the predictions, metrics, scatter dashboard, and paired
LightGBM comparison. The command rejects artifacts with a different test split
or power-area normalization.

## Compare

```bash
.venv/bin/python -m neural_building_emulator_refactor.compare \
  --artifact-dir output/run_a/artifacts/q_to_t_lstm/selected \
  --artifact-dir output/run_b/artifacts/probabilistic/selected \
  --dataset neural_building_emulator/tessin_results.parquet/all_hp \
  --output-dir output/neural_building_emulator_refactor/comparison \
  --profile-source test \
  --num-scenarios 100
```

This writes one `model_comparison.csv`. Compare models within the same task;
the Q-to-T and closed-loop problems expose different information and are not a
fair head-to-head measure of the same predictive problem.

### Fair structured-versus-LSTM comparison

The frozen comparison recipe is
[`configs/fair_structured_vs_probabilistic_lstm.json`](configs/fair_structured_vs_probabilistic_lstm.json).
It evaluates four arms: deterministic structured, deterministic vanilla LSTM,
probabilistic structured, and probabilistic vanilla LSTM. The probabilistic structured
arm reproduces the best setup immediately before the experimental Markov
`Qroom` gate: continuous `Qroom`, asymmetric-Markov `Pel`, setpoint-shock
traces, positive-leaky thermal response, 24 training particles, and the same
energy/variogram/soft-Best-of-K and HP-emission losses. Every arm receives the
same inputs, metadata, split, windows, target normalization, optimizer budget,
and annual scoring protocol. The vanilla arms predict all outputs through an
unconstrained linear readout and receive no HP-specific engineered features or
losses. Deterministic models are checkpointed using joint
normalized `Tin/Qroom/Pel` error over every test window.

Reuse the already trained probabilistic structured model and train the other
three arms:

```bash
caffeinate .venv/bin/python -m neural_building_emulator_refactor.fair_probabilistic_lstm_comparison \
  --structured-artifact-dir output/neural_building_emulator_refactor/setpoint_shock_independent_new_dataset_markov/artifacts/closed_loop_hp_contracting_probabilistic/selected \
  --output-dir output/neural_building_emulator_refactor/fair_structured_vs_probabilistic_lstm
```

Omit `--structured-artifact-dir` to retrain all four arms. Each arm also has a
specific `--*-artifact-dir` option for resuming a partial comparison. The runner
refuses both mismatched model types and artifacts with different ordered test
splits, then evaluates complete
autonomous annual rollouts. Probabilistic arms use 100 linked scenarios;
deterministic arms contribute point predictions and deterministic flexibility
metrics. The comparison table includes annual `Tin/Qroom/Pel` RMSE and bias,
plus upward/downward regression and direct-ratio `F_H` RMSE. Both flexibility
estimators are computed from the same trajectories. Outputs include per-arm
traces, `model_comparison.csv`, `flexibility_coverage_comparison.csv`,
`temporal_trace_calibration_comparison.csv`, and
`fair_comparison_dashboard.html`.
Probabilistic arms additionally report HP start/stay-on reliability, active-run
duration distributions, mode autocorrelation, annual-electric-energy rank
calibration, and Pel rank calibration conditioned on setpoint-shock age and
direction, previous observed HP mode, and thermostat deficit. These are shown
in `pel_dynamics_comparison.html` and summarized in `model_comparison.csv`.
Recorded ventilation is used throughout so all models receive exactly the same
exogenous trajectory; no architecture-specific EnergyPlus ventilation feedback
is used.

## Fast Ablation KPIs

The full flexibility-coverage analysis is intentionally a final evaluation: it
requires many complete probabilistic scenarios and fits an event-study model to
each one. During architecture ablations, use the mean-trajectory scorecard:

The formulas, interpretation, initial characterization results, and recommended
selection protocol are documented in [FAST_ABLATION_KPIS.md](FAST_ABLATION_KPIS.md).

```bash
.venv/bin/python -m neural_building_emulator_refactor.fast_ablation_kpis \
  --artifact-dir output/run/artifacts/closed_loop_hp_contracting_probabilistic/selected \
  --dataset neural_building_emulator/tessin_results.parquet/all_hp \
  --output-dir output/run/fast_kpis \
  --profile-source test \
  --max-profiles 100 \
  --num-eval-particles 4
```

It reports four complementary kinds of fidelity:

- `total_nrmse`: equal-channel RMSE after division by each training target
  scale. This is the primary scalar ranking metric.
- normalized Wasserstein distance: marginal operating-range fidelity.
- ACF mean absolute error and increment-spectrum Jensen-Shannon distance:
  temporal memory, cycling, and frequency-content fidelity.
- a raw 3-hour flexibility response: post-event mean `Pel` minus the equally
  long pre-event mean at clean `Tset` discontinuities. Upward and downward
  energy gains are the mean event response divided by the setpoint change and
  multiplied by 3 hours. No controls, regression, or scenario calibration are
  involved.

The raw flexibility KPI is a fast model-selection proxy, not a replacement for
the controlled event-study and probabilistic coverage analysis. The evaluator
also writes event-level response data and a Spearman heatmap of simulated
profile descriptors. This makes it possible to check whether an apparent model
improvement is specific to point error, temporal morphology, or thermostat
response.

Compare several ablations on exactly the same split with one command:

```bash
.venv/bin/python -m neural_building_emulator_refactor.fast_compare \
  --artifact-dir output/ablation_a/artifacts/closed_loop_hp_contracting_probabilistic/selected \
  --artifact-dir output/ablation_b/artifacts/closed_loop_hp_contracting_probabilistic/selected \
  --dataset neural_building_emulator/tessin_results.parquet/all_hp \
  --output-dir output/fast_ablation_comparison \
  --max-profiles 100 \
  --num-eval-particles 4
```

The comparison refuses artifacts with different ordered split IDs. It writes
the detailed dashboard for each artifact and one
`fast_ablation_comparison.csv` with mean, median, and p90 scores.

The measured profile-count/stride scaling study and the recommended fast
screening budget are documented in
[DETERMINISTIC_SCALING_STUDY.md](DETERMINISTIC_SCALING_STUDY.md). The study
runner keeps a fixed holdout and nested training subsets, saves every model,
and resumes completed cells.

## Deterministic Optuna Study

Use the deterministic contracting closed-loop model to select structural choices
and deterministic loss balance before paying for probabilistic tuning:

```bash
.venv/bin/python -m neural_building_emulator_refactor.optuna_deterministic_study \
  --dataset neural_building_emulator/tessin_results.parquet/all_hp \
  --output-dir output/neural_building_emulator_refactor/optuna_deterministic_study_8ep \
  --max-profiles 1000 \
  --validation-profiles 30 \
  --final-test-profiles 30 \
  --num-trials 60 \
  --min-trials-per-architecture 3 \
  --epochs 8 \
  --sequence-length 960 \
  --stride 8096 \
  --confirmation-top-k 3 \
  --confirmation-seeds 13 29 47
```

The objective combines normalized trajectory error, temperature temporal
dynamics, and the raw three-hour thermostat-response error. It writes an
interactive study dashboard, balanced architecture reports, a ranked top-k,
paired multi-seed confirmation, and a final untouched-cohort comparison with
annual traces. The split protocol, equations, weight justification, search
space, and interpretation limits are documented in
[OPTUNA_DETERMINISTIC_STUDY.md](OPTUNA_DETERMINISTIC_STUDY.md).

For the longer follow-up after the eight-epoch search, pass
`--epochs 15 --fixed-learning-rate 8.5e-4` and use a fresh output directory.
The fixed rate is applied consistently to the baseline, search trials, and
multi-seed confirmation runs.

## Original-Six Parity

The six original state-space model kinds have registry adapters. Their exact
training and artifact parity can be retested with:

```bash
.venv/bin/python -m neural_building_emulator_refactor.parity_original_six \
  --dataset neural_building_emulator/tessin_results.parquet/all_hp \
  --output-dir output/neural_building_emulator_refactor/original_six_parity \
  --max-profiles 100 \
  --test-fraction 0.1 \
  --seed 41 \
  --sequence-length 96 \
  --stride 40000 \
  --batch-size 32 \
  --prob-particles 2 \
  --prob-eval-particles 2
```

This independently trains each original model through the direct legacy path
and the registry adapter. It compares selected model leaves, scalers, split
IDs, checkpoint metrics, and one reloaded full-profile prediction with zero
tolerance. Detailed logs and `parity_results.csv` are retained in the output
directory. The large stride intentionally gives one 96-step window per
building: all 100 selected buildings participate while the purpose remains
behavioral parity rather than model-quality benchmarking.

## Force-Off Policy Transfer

Closed-loop models can also be trained jointly on thermostat-identification and
force-off policies:

```bash
.venv/bin/python -m neural_building_emulator_refactor.train \
  --model closed_loop_hp_contracting_probabilistic \
  --dataset data/control=setpoint_identification_dhw_fixed \
  --additional-dataset data/control=force_off \
  --output-dir output/neural_building_emulator_refactor/mixed_policy
```

`--additional-dataset` may be repeated. The split remains building-level: every
policy trajectory for an EGID is assigned to the same train or test side. If a
source contains `hp_force_off_requested`, that signal is included in the saved
input schema and is zero-filled for sources that omit it.

For structured models, force-off is a causal generation gate:
`Pel_SH = (1 - force_off) * Pel_SH_unrestricted`. The signal is hidden from the
learned HP-demand and thermal networks, and it does not directly suppress
`Qreq`, buffer discharge, `Qroom`, or temperature. For deterministic and
probabilistic LSTM models it is passed as an ordinary exogenous input, allowing
the unstructured baseline to learn its own response.

### Optional Hydronic Delivery Innovation

For the probabilistic contracting model, `--prob-hydronic-innovation` adds a
per-step, zero-mean innovation to the latent room-heat request before the
buffer draw. Its learned scale is bounded by
`--prob-hydronic-innovation-cap-w-m2` (default `20 W/m2`) and shares the
existing process-noise scale parameter. A positive-leaky delivery-request
state with fixed `--prob-hydronic-innovation-time-constant-hours` (default
`0.5 h`) represents hydronic transport. The actual delivery remains
nonnegative and energy-limited:

`Qroom_t = buffer_draw(E_t, COP_t * Pel_t, max(Qrequest_t + epsilon_Q,t, 0))`.

Consequently, the innovation can express unobserved hydronic timing and
delivery variation, but cannot discharge an empty buffer or create energy.
The option is disabled by default and does not change older artifacts.

Evaluate a probabilistic contracting model trained under thermostat excitation
on the out-of-policy HP force-off simulations with:

```bash
.venv/bin/python -m neural_building_emulator_refactor.force_off_policy_evaluation \
  --artifact-dir output/my_run/artifacts/closed_loop_hp_contracting_probabilistic/selected \
  --dataset data/control=force_off \
  --output-dir output/my_run/force_off_policy \
  --profile-source test \
  --num-eval-particles 16 \
  --ventilation-rollout-mode eplus_rule
```

The intervention is applied at the HP generation boundary: forced-off steps
set predicted `Pel_SH` to zero before the buffer update, while room-heat demand
and downstream thermal states remain active. The report compares this causal
rollout with the same structured model ignoring the policy and writes annual
profiles, force-off onset responses, and metrics split between forced-off and
unrestricted periods. `--unstructured-artifact-dir` optionally adds an LSTM
artifact as an OOD baseline. Older LSTM artifacts do not receive the command;
mixed-policy artifacts whose saved schema contains `hp_force_off_requested` do.

## Migration Boundary

The complete structured-versus-vanilla, deterministic-versus-probabilistic,
LightGBM, and held-out calibration protocol is documented in
[CALIBRATED_MODEL_BENCHMARK.md](CALIBRATED_MODEL_BENCHMARK.md). It provides one
resumable command and one consolidated HTML report while preserving all raw
per-building and reliability tables.

The six state-space registry entries deliberately delegate training and
sampling to `neural_building_emulator`. This avoids silently changing equations
while the refactor matures. New orchestration, artifacts, LSTMs, prediction, and
scoring live entirely in this folder. A later migration can move one legacy
component at a time behind the same interfaces and verify artifact-level parity.
