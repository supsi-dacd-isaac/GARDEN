# Neural Building Emulator Refactor

This package is a parallel comparison pipeline. It does not replace or modify
`neural_building_emulator`; the established state-space equations and trainer
remain available there. The refactor adds a small registry, common artifacts,
two recurrent baselines, and one reload-and-score interface.

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

The LSTMs are intentionally unstructured controls. Static metadata initializes
their hidden and cell states. During autonomous rollout, the previous predicted
output is fed back: `Tin` for Q-to-T and `[Tin,Qroom,Pel]` for closed loop. They
have no stability, positivity, energy-conservation, or monotonicity constraints.

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
and rank-histogram CSV files. Temperature uses all timestamps; the two heating
signals use only timestamps when space heating is available so the diagnostic
is not dominated by structurally zero off-season predictions.

### Metadata-only flexibility benchmark

To test whether the closed-loop emulator is needed when only the final
building-level flexibility index is required, fit LightGBM directly from the
same 15 static metadata fields to EnergyPlus event-study KPIs:

```bash
.venv/bin/python -m neural_building_emulator_refactor.metadata_flexibility_lightgbm \
  --artifact-dir output/.../artifacts/closed_loop_hp_contracting_probabilistic/selected \
  --dataset neural_building_emulator/tessin_results.parquet/all_hp \
  --output-dir output/neural_building_emulator_refactor/metadata_flexibility_lightgbm \
  --emulator-kpi-csv output/.../flexibility_event_study_profile_kpis.csv \
  --horizons-hours 0.5 1 2 3 \
  --controls full
```

The artifact supplies the exact building-level train/test split. KPI targets
are calculated from EnergyPlus only; LightGBM receives no time-series inputs,
weather, or emulator predictions. Separate models are fitted for upward and
downward flexibility at each horizon. Outputs include reloadable LightGBM text
models, per-building predictions, metrics, paired bootstrap comparisons with
the emulator, feature importance, and interactive held-out scatter/score
dashboards.

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

## Migration Boundary

The six state-space registry entries deliberately delegate training and
sampling to `neural_building_emulator`. This avoids silently changing equations
while the refactor matures. New orchestration, artifacts, LSTMs, prediction, and
scoring live entirely in this folder. A later migration can move one legacy
component at a time behind the same interfaces and verify artifact-level parity.
