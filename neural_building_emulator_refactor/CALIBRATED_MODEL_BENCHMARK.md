# Calibrated Four-Model Benchmark

`calibrated_model_benchmark.py` is the end-to-end comparison used to answer
three separate questions without changing datasets or splits between models:

1. Does the structured model improve autonomous annual trajectory fidelity?
2. Does either probabilistic sequence model represent uncertainty and temporal
   behavior faithfully?
3. Is full emulation more useful for flexibility KPIs than a metadata-only
   LightGBM predictor?

## Compared Models

The runner trains four matched models:

| Arm | Model |
|---|---|
| `deterministic_structured` | Contracting closed-loop HP state-space emulator |
| `deterministic_lstm` | Vanilla deterministic autoregressive LSTM |
| `probabilistic_structured` | Probabilistic contracting closed-loop HP emulator |
| `probabilistic_lstm` | Vanilla probabilistic autoregressive LSTM |

The vanilla LSTMs use the same observed inputs and predict the same three
outputs, but have no COP/buffer equations, positivity constraints, thermostat
monotonicity, contraction, or engineered HP switching mechanism. The checked-in
default recipe is
`configs/calibrated_four_model_benchmark.json`. It records the current
pre-Markov-Q structured recipe and the matching LSTM settings.

Training uses recorded EnergyPlus ventilation. Annual scoring now defaults to
`eplus_rule`: each closed-loop arm regenerates window flow sequentially from
the EnergyPlus schedule, outdoor temperature, and its own predicted indoor
temperature. This avoids feeding the scored model a ventilation trace driven
by the simulated temperature. The mode is inference-only and does not require
retraining saved models. Changing the recipe mode invalidates cached score
requests; reports made from old score tables remain recorded-ventilation
results until the `score`, `calibrate`, and `report` stages are rerun.

For probabilistic structured artifacts, `score.py --hp-scenario-mode expected`
and `--hp-scenario-mode bernoulli` score the same checkpoint with different HP
emission rollouts. The vanilla probabilistic LSTM has no separate HP emission,
so that flag does not change its trajectory sampler.

## Four-Way Split

The selected SH-HP buildings are divided once into disjoint cohorts:

- `train`: gradient updates and LightGBM fitting;
- `validation`: neural checkpoint selection and LightGBM early stopping;
- `calibration`: residual-normalized score calibration only;
- `test`: final metrics, reliability curves, and paired comparisons only.

The default proportions are 70/10/10/10. Exact profile IDs and a split
fingerprint are stored in `split_manifest.json`. Every model is checked against
that manifest. This prevents checkpoint selection, uncertainty calibration,
and final reporting from silently sharing buildings.

## LightGBM And Calibration

Separate metadata-only LightGBM quantile regressors are fitted for every
combination of:

- intervention policy: randomized setpoint or HP force-off, when the force-off
  dataset is supplied;
- flexibility definition: regression-adjusted response or direct `F`;
- direction: upward, downward, or force-off;
- horizon: 0.5, 1, 2, or 3 hours.

The benchmark always computes both the direct and regression-adjusted
definitions. They use one common long-form schema with
`intervention_type` in `{tset_up, tset_down, force_off}` and `estimator` in
`{direct_ratio, regression}`.

For a setpoint event, `delta_P_H` is mean post-event minus mean pre-event HP
power. The direct definitions use positive useful-effect signs:

```text
F_H(tset_up)   = H * sum(delta_P_H) / sum(delta_Tset),  delta_Tset > 0
F_H(tset_down) = H * sum(delta_P_H) / sum(delta_Tset),  delta_Tset < 0
```

Both are in `Wh/(m2 K)` because numerator and denominator have matching signs.
For a persistent force-off onset, the useful effect is shed power:

```text
F_H(force_off) = H * sum(P_pre_H - P_post_H) / sum(force_off_dose)
```

It is in `Wh/(m2 intervention)`, not `Wh/(m2 K)`. The force-off regression
estimator compares persistent onsets with stable no-intervention windows while
adjusting for pre-event power, indoor state, weather, gains, ventilation, and
calendar features. The setpoint regression estimator retains separate linear
up/down coefficients and the configured event controls.

The raw LightGBM scenario distribution is a Gaussian location-scale
approximation derived from the 0.16, 0.50, and 0.84 quantile models. This is an
explicit approximation, not a claim that the KPI is intrinsically Gaussian.

Both probabilistic emulators and LightGBM are also calibrated with a
residual-normalized score (RNS). For calibration building `i`, the signed score
is

```text
r_i = (y_i - median(samples_i)) / max(IQR(samples_i) / 1.349, scale_floor).
```

The empirical distribution of `r_i` is fitted separately for each model, KPI,
horizon, and direction. Test scenarios are transformed with that one fitted
score distribution. All quantiles therefore come from one ordered empirical
CDF and cannot cross. RNS changes scalar KPI distributions only; it does not
alter annual `Tin`, `Qroom`, or `Pel` trajectories.

## Reported Metrics

Annual autonomous rollouts report:

- RMSE, MAE, bias, and normalized RMSE for `Tin`, `Qroom`, and `Pel`;
- normalized Wasserstein distance;
- autocorrelation mismatch;
- increment-spectrum Jensen-Shannon distance;
- temporal empirical-rank reliability for all three signals;
- `Qroom` on/off Brier score, log loss, ECE, and positive-value rank coverage;
- HP start/stop reliability, conditional calibration, mode ACF, run durations,
  and annual electrical-energy calibration.

Both flexibility definitions report point RMSE, NRMSE, MAE, bias, R2, rank
correlations, and top-decile overlap. Probabilistic KPI distributions report
CRPS, reliability curves, 50/80/90/95 percent coverage, and interval widths,
before and after RNS calibration.

Paired bootstrap comparisons use identical held-out buildings. Their reported
error difference is `method_a - method_b`, so a negative value favors method A.

## Run

Run the complete default benchmark:

```bash
caffeinate .venv/bin/python -m neural_building_emulator_refactor.calibrated_model_benchmark
```

Use a separate output directory for each experiment:

```bash
caffeinate .venv/bin/python -m neural_building_emulator_refactor.calibrated_model_benchmark \
  --config neural_building_emulator_refactor/configs/calibrated_four_model_benchmark.json \
  --output-dir output/neural_building_emulator_refactor/calibrated_four_model_benchmark
```

For the mixed thermostat-identification plus force-off experiment, use the
metadata-alpha Option C recipe and add the policy dataset explicitly:

```bash
caffeinate .venv/bin/python -m neural_building_emulator_refactor.calibrated_model_benchmark \
  --config neural_building_emulator_refactor/configs/calibrated_four_model_benchmark_option_c_metadata_alpha.json \
  --additional-dataset data/control=force_off \
  --contracting-temperature-feedback-gradient-scale 0.05 \
  --output-dir output/neural_building_emulator_refactor/calibrated_four_model_benchmark_mixed_policy \
  --max-profiles 1000 \
  --epochs 15 \
  --no-resume
```

All four neural arms receive both policy trajectories and retain a common
building-level split. Annual reliability, flexibility KPIs, and LightGBM are
evaluated on the primary randomized-setpoint dataset. The same run evaluates
calibration and test cohorts from the force-off dataset under
`force_off_policy/<split>/`. The sequential policy-transfer dashboard remains
at `force_off_policy/test/force_off_policy_dashboard.html`; LightGBM does not
produce time series, but its direct and regression-adjusted force-off KPI
distributions are included in scalar-KPI calibration.

The benchmark selects three held-out setpoint-policy buildings and three
different force-off-policy buildings when the test split is large enough. For
both probabilistic neural models it writes full-year Tin/Qroom/Pel plots with
the scenario mean, 95% interval, and the closest and worst complete scenario
for each output. Setpoint examples live under
`scoring/test/<probabilistic-arm>/`; force-off examples live under
`force_off_policy/test/probabilistic_traces/`. The selected paths are indexed
by `reports/full_year_traces.html` and
`reports/full_year_trace_manifest.json`.

LightGBM is policy-aware through separate fitted models: setpoint-modulation
and force-off targets are never pooled into one regressor. Policy identity is
therefore constant within each fit and is recorded in each model manifest,
rather than added as a redundant constant feature. Both policies also produce
structured-versus-LSTM temporal reliability and ensemble-rank dashboards in
`reports/setpoint_temporal_trace_calibration.html` and
`reports/force_off_temporal_trace_calibration.html`.

The stages are independently resumable:

```bash
.venv/bin/python -m neural_building_emulator_refactor.calibrated_model_benchmark \
  --output-dir output/neural_building_emulator_refactor/calibrated_four_model_benchmark \
  --stages score lightgbm force_off calibrate report
```

Use `--no-resume` to replace completed work. Existing artifacts can be supplied
with the four `--*-artifact-dir` options. The runner still checks their split
against the benchmark manifest.

## Outputs

The main entry point is `reports/benchmark_dashboard.html`. CSV files beside it
retain the underlying values, including per-building flexibility predictions,
paired bootstrap results, rank histograms, signal reliability, HP dynamics,
and raw/RNS probabilistic scores. The unified
`reports/intervention_flexibility_kpis.csv` and scenario companion contain
`F(tset+)`, `F(tset-)`, and `F(force_off)` in one schema. Setpoint calibration
is shown in `calibration/kpi_calibration.html`; force-off calibration is shown
separately in `calibration/force_off_kpi_calibration.html` so unlike units are
not mixed. Arm-specific annual plots and diagnostic HTML files remain under
`scoring/test/<arm>/` for detailed inspection.

New LSTM runs also write `artifacts/<model>/gradient_updates_*.csv` under each
arm's training output. Each minibatch row contains the pre-clipping gradient
norm, the clip and skip thresholds, loss, and accept/skip reason. Rows are
flushed during training, so an interrupted run retains its diagnostic history.
The completed benchmark predating this change has no recoverable LSTM gradient
history in its saved artifacts.
