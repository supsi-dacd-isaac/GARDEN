# Deterministic Model-Selection Study

## Purpose

`optuna_deterministic_study.py` selects the structure and main training weights
of the deterministic contracting closed-loop HP emulator. It deliberately does
not tune probabilistic-only choices such as particle count, process noise,
energy-score weight, variogram weight, or IRES weight. Those choices cannot be
identified from a deterministic model and need a smaller second-stage study
after the deterministic structure has been fixed.

The original screening study used the faster setting supported by the scaling
experiment: 960-step rollouts, stride 8096, eight epochs, and all available
training buildings. Every trial uses exactly the same training profiles and
validation profiles.

Within each trial, the selected checkpoint minimizes the validation-window score

\[
J_{\mathrm{window}}=\sqrt{\frac{1}{3}\left[
\left(\frac{\mathrm{RMSE}_T}{s_T}\right)^2+
\left(\frac{\mathrm{RMSE}_Q}{s_Q}\right)^2+
\left(\frac{\mathrm{RMSE}_P}{s_P}\right)^2
\right]}.
\]

This evaluation is already performed every epoch and therefore adds no rollout
cost. Q-to-T models continue to select checkpoints using temperature RMSE.

## Data Split

The script creates and persists three disjoint profile sets:

1. A training set used to fit every trial.
2. A fixed validation set used to calculate the Optuna objective.
3. An untouched final-test set excluded from both training and validation.

The IDs are saved in `profile_split.json`. Reusing the output directory resumes
the same study and split. Changing a study-defining argument requires a new
output directory. The final-test IDs should only be scored after the search and
the planned multi-seed confirmation have selected a model; otherwise they stop
being an honest final test. The runner now enforces this order automatically:
the final cohort is not evaluated until all requested search trials and the
validation-only multi-seed confirmation have completed.

## Objective

The selected objective is

\[
J = 0.50 J_{\mathrm{trajectory}}
  + 0.20 J_{\mathrm{temporal}}
  + 0.30 J_{\mathrm{flex}}.
\]

The three terms are divided by the corresponding result of the current
hand-selected baseline trial. The baseline therefore has
`J_trajectory = J_temporal = J_flex = 1` and total objective 1. A value below 1
means improvement over the baseline under the selected trade-off.

### Trajectory fidelity: 0.50

\[
J_{\mathrm{trajectory}} =
\frac{\operatorname{NRMSE}_{T,Q,P}}
     {\operatorname{NRMSE}^{\mathrm{baseline}}_{T,Q,P}},
\]

where

\[
\operatorname{NRMSE}_{T,Q,P} =
\sqrt{\frac{1}{3N}\sum_{t,j}
\left(\frac{\hat y_{t,j}-y_{t,j}}{s_j}\right)^2}
\]

and `j` indexes `Tin`, `Qroom`, and `Pel`. The scales `s_j` are computed once
from the fixed validation targets, so a trial cannot improve its score by
changing its own scaler.

This receives half the weight because autonomous trajectory fidelity is a
necessary condition for a useful emulator. A model with a plausible thermostat
response but wrong annual temperatures or energy flows is not acceptable. It
also protects against optimizing only a small set of setpoint events. The weight
is not larger than 0.50 because pointwise `Pel` is intrinsically noisy and
compressor phase is not the final scientific target.

### Temperature temporal dynamics: 0.20

\[
J_{\mathrm{temporal}} = \frac{1}{2}
\left(
\frac{D_{\mathrm{ACF}}}{D^{\mathrm{baseline}}_{\mathrm{ACF}}}
+
\frac{D_{\mathrm{PSD}}}{D^{\mathrm{baseline}}_{\mathrm{PSD}}}
\right).
\]

`D_ACF` is the mean absolute difference between simulated and emulated
temperature autocorrelations at lags 1, 2, 4, 8, 12, 24, 48, and 96 samples.
`D_PSD` is the Jensen-Shannon distance between normalized power spectra of the
first temperature differences.

This term catches failures that RMSE can hide: excessive smoothing, unrealistic
high-frequency temperature motion, and incorrect thermal memory. Its weight is
kept at 0.20 because it is a diagnostic of trajectory shape, not an independent
physical target, and it partially overlaps temperature RMSE. ACF and spectrum
receive equal weight inside the term because one measures lag-domain memory and
the other measures where increment variance lies in frequency.

### Setpoint-response fidelity: 0.30

For each clean setpoint discontinuity, the evaluator computes

\[
\Delta \bar P_{k,3h} =
\frac{1}{N_H}\sum_{j=0}^{N_H-1}P_{t_k+j}
-
\frac{1}{N_H}\sum_{j=1}^{N_H}P_{t_k-j}.
\]

The score is

\[
J_{\mathrm{flex}} =
\frac{
\operatorname{RMSE}(\Delta\bar P^{\mathrm{emu}}_{k,3h},
                     \Delta\bar P^{\mathrm{sim}}_{k,3h})/s_P}
{
\operatorname{NRMSE}^{\mathrm{baseline}}_{\mathrm{flex}}}.
\]

This receives 0.30 because reproducing the conditional electrical response to a
randomized thermostat intervention is the main downstream use of the closed-loop
emulator. It is high enough that a model cannot win merely by learning mean
seasonal trajectories, but lower than trajectory fidelity because it is a noisy
three-hour proxy rather than the final controlled event-study estimator or a
probabilistic calibration score.

### Interpreting the weights

Baseline normalization makes the weights operational. Holding the other terms
fixed, a 10% relative improvement changes the objective by:

- `0.05` for trajectory fidelity;
- `0.03` for intervention response;
- `0.02` for temporal dynamics.

Thus the model may accept a small pointwise-accuracy loss for a substantial
causal-response improvement, but not an arbitrarily inaccurate trajectory. Raw
metrics with different units would not permit this interpretation.

The selected weights are still a scientific preference, not a theorem. The
study therefore recalculates every trial under accuracy-heavy (65/15/20) and
flexibility-heavy (35/20/45) weights. `objective_weight_sensitivity.html` shows
whether the leading trials and architecture families remain competitive. A
candidate whose rank collapses under a modest reweighting should not be treated
as a robust winner.

## Search Space

The study varies:

- unconstrained versus monotone thermostat demand;
- unconstrained versus positive-leaky `Qroom -> Tin` response;
- inclusion of controller calendar features;
- thermal state dimension, controller state dimension, hidden dimension, and
  input-encoder dimension;
- learning rate and maximum temperature increment;
- relative deterministic losses on `Qroom`, `Pel`, and HP mode;
- positive-leaky time constants and gain cap when that structure is active.

The contraction bound, bounded states and outputs, availability gate, leaky
equilibrium update, COP cap, and TBPTT length remain fixed. These are safety and
experimental-protocol choices rather than ordinary capacity knobs.

Passing `--fixed-learning-rate` removes learning rate from the search and uses
that value for the mandatory baseline, every trial, and all confirmation
replicas. It is consequently omitted from parameter-importance plots. This is
useful for a longer follow-up study after a shorter search has already localized
a suitable learning-rate range.

The first trial is the current hand-selected baseline. Subsequent trials use
Optuna's multivariate grouped TPE sampler after startup random trials. Before
free TPE exploration, the runner queues a balanced block with at least three
trials for each of the eight combinations of thermostat mode, `Qroom -> Tin`
mode, and controller-calendar inclusion. This prevents an early noisy family
from consuming almost the entire budget. No
early-trial pruning is used: eight-epoch structural comparisons can be
non-monotone, so pruning would risk systematically rejecting slower-starting
architectures. Optuna references: [TPE sampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html),
[fANOVA](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.importance.FanovaImportanceEvaluator.html),
and [PED-ANOVA](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.importance.PedAnovaImportanceEvaluator.html).

## Run

```bash
.venv/bin/python -m neural_building_emulator_refactor.optuna_deterministic_study \
  --dataset neural_building_emulator/tessin_results.parquet/all_hp \
  --output-dir output/neural_building_emulator_refactor/optuna_deterministic_study_8ep \
  --max-profiles 1000 \
  --validation-profiles 30 \
  --final-test-profiles 30 \
  --num-trials 60 \
  --startup-trials 16 \
  --min-trials-per-architecture 3 \
  --epochs 8 \
  --sequence-length 960 \
  --stride 8096 \
  --batch-size 32 \
  --train-eval-max-windows 64 \
  --report-top-k 10 \
  --confirmation-top-k 3 \
  --confirmation-seeds 13 29 47 \
  --final-profile-plots 4 \
  --seed 13
```

### Fifteen-epoch fixed-rate follow-up

The completed eight-epoch study placed nearly all leading trials near the upper
end of the learning-rate range, while every confirmed candidate still selected
epoch eight. A clean follow-up therefore fixes the rate at `8.5e-4`, close to
the multi-seed winner, and extends all candidates to 15 epochs:

```bash
caffeinate -is .venv/bin/python -m neural_building_emulator_refactor.optuna_deterministic_study \
  --dataset neural_building_emulator/tessin_results.parquet/all_hp \
  --output-dir output/neural_building_emulator_refactor/optuna_deterministic_study_15ep_fixed_lr \
  --study-name contracting_hp_deterministic_15ep_fixed_lr \
  --max-profiles 1000 \
  --validation-profiles 30 \
  --final-test-profiles 30 \
  --num-trials 60 \
  --startup-trials 16 \
  --min-trials-per-architecture 3 \
  --epochs 15 \
  --fixed-learning-rate 8.5e-4 \
  --sequence-length 960 \
  --stride 8096 \
  --batch-size 32 \
  --train-eval-max-windows 64 \
  --report-top-k 10 \
  --confirmation-top-k 3 \
  --confirmation-seeds 13 29 47 \
  --final-profile-plots 4 \
  --seed 13
```

Use a fresh output directory as above. The baseline normalization is recomputed
at the fixed rate, and checkpoint selection still retains the best validation
epoch rather than forcing the epoch-15 weights.

`--num-trials` is a target total, so rerunning the command resumes the journal
instead of repeating completed trials. It is part of the persisted protocol and
cannot be increased after the run, because the original completion may already
have exposed the final-test results. The default confirmation compares the
three leading non-baseline configurations with the baseline over three seeds.
The search-seed artifacts are reused, so this adds eight trainings rather than
twelve. Each candidate is normalized against the baseline trained with the
same seed. The configuration with the lowest mean validation objective is then
selected. Only that confirmed winner and the baseline are evaluated on the
untouched final cohort.

## Outputs

- `optuna_study_dashboard.html`: optimization history, accuracy/flexibility and
  temporal trade-offs, architecture families, importance, and runtime.
- `objective_weight_sensitivity.html`: objective and rank changes under three
  weight policies.
- `optuna_parameter_slices.html`: one-dimensional parameter slices.
- `optuna_parallel_coordinates.html`: interactions among parameters common to
  all conditional branches.
- `trials.csv`: parameters, objective components, and physical KPIs.
- `top_trials.csv` and `top_trials.json`: the ranked top configurations rather
  than only a nominal winner.
- `architecture_summary.csv`: family-level medians and best scores.
- `parameter_importance.csv`: PED-ANOVA and fANOVA estimates.
- `best_trial.json`: Optuna's nominal single best trial, retained for
  compatibility; scientific selection should use the confirmation outputs.
- `architecture_quota_plan.json`: the balanced architecture allocation.
- `confirmation/confirmation_runs.csv`: per-seed validation results against
  same-seed baselines.
- `confirmation/confirmation_summary.csv` and
  `confirmation_dashboard.html`: mean/spread and ranking of the confirmed
  candidates.
- `confirmation/confirmed_selection.json`: the validation-only model choice.
- `final_test/final_test_runs.csv`, `final_test_summary.csv`, and
  `final_test_comparison.html`: the one-time untouched-cohort comparison.
- `final_test/full_year_traces/{baseline,winner}/`: representative annual
  traces using the seed nearest each configuration's mean validation score.
- `trials/trial_XXXX/`: complete log, config, artifact, and fast-KPI reports for
  every trial.

Parameter importance is exploratory, not causal. Conditional search spaces,
parameter interactions, and a finite TPE sample can all affect the estimates.
The generated multi-seed and untouched-test reports, rather than
`best_trial.json` alone, are the basis for a final structural claim.
