# Deterministic Scaling Study

## Question

This study estimates how training cost and fixed-holdout performance change
with the number of buildings and the rollout-window stride. It uses the two
deterministic counterparts of the models needed in the planned ablation:

- `q_to_t_deterministic_ss`
- `closed_loop_hp_contracting_deterministic`

The purpose is budget selection, not a final model comparison. Three epochs
are enough to expose poor data regimes, but the absolute scores are not the
expected converged performance of either model.

## Controlled Design

- Dataset: `neural_building_emulator/tessin_results.parquet/all_hp`
- Q-to-T candidates: 1,000 profiles
- Closed-loop HP candidates after the common HP filter: 963 profiles
- Fixed test holdout: the same 20 HP-eligible profile IDs in every run
- Nested training sets: the 100-profile set is contained in the 300-profile
  set, which is contained in the full set
- Sequence length: 960 steps, or 10 days at 15-minute resolution
- Epochs: 3
- Rotating window starts: enabled
- Closed-loop TBPTT: 32 steps
- Per-epoch train evaluation: fixed 64-window subset
- Full train evaluation and plots during training: disabled
- Fast KPI normalization: fixed standard deviations computed from the common
  holdout, rather than the run-specific training scaler
- Seed: 13

Compact mode evaluates profile scaling at stride 4,048, stride scaling at 300
profiles, all three strides at full population, and the compute-matched pair:

```text
300 profiles / stride 2024  ~=  full population / stride 8096
```

Both have about 4,700-4,900 training windows.

## Command

```bash
.venv/bin/python -m neural_building_emulator_refactor.deterministic_scaling_study \
  --dataset neural_building_emulator/tessin_results.parquet/all_hp \
  --output-dir output/neural_building_emulator_refactor/deterministic_scaling_study \
  --models q_to_t_deterministic_ss closed_loop_hp_contracting_deterministic \
  --profile-counts 100 300 1000 \
  --strides 2024 4048 8096 \
  --design compact \
  --epochs 3 \
  --sequence-length 960 \
  --batch-size 32 \
  --holdout-profiles 20 \
  --train-eval-max-windows 64 \
  --seed 13 \
  --quality-tolerance 0.10
```

The command is resumable. Completed run directories containing
`study_result.json` are not retrained.

## Results

### Q-to-T deterministic

| Profiles | Stride | Windows | Epoch min | Wall min | Tin NRMSE |
|---:|---:|---:|---:|---:|---:|
| 100 | 4,048 | 720 | 0.023 | 0.239 | 0.940 |
| 300 | 2,024 | 4,760 | 0.053 | 0.545 | 0.612 |
| 300 | 4,048 | 2,520 | 0.033 | 0.464 | 0.748 |
| 300 | 8,096 | 1,400 | 0.027 | 0.430 | 0.764 |
| 1,000 | 2,024 | 16,660 | 0.133 | 1.576 | 0.476 |
| 1,000 | 4,048 | 8,820 | 0.077 | 1.353 | 0.538 |
| 1,000 | 8,096 | 4,900 | 0.050 | 1.199 | 0.571 |

At almost equal window count, 1,000/8,096 beats 300/2,024 (`0.571` versus
`0.612`). Metadata/building diversity is therefore more valuable than dense
within-profile sampling for this task. The Q-to-T deterministic model is cheap
enough that retaining all profiles and stride 2,024 is still the preferred
deterministic setting. For a particle-heavy probabilistic screening run,
stride 4,048 is a defensible compromise, but it is 13% worse than the dense
reference after three epochs.

### Contracting closed-loop HP deterministic

| Profiles | Stride | Windows | Epoch min | Wall min | Total NRMSE | Tin | Qroom | Pel |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 4,048 | 720 | 0.133 | 0.663 | 0.907 | 0.867 | 0.818 | 0.942 |
| 300 | 2,024 | 4,760 | 0.457 | 1.876 | 0.599 | 0.457 | 0.586 | 0.638 |
| 300 | 4,048 | 2,520 | 0.260 | 1.285 | 0.610 | 0.464 | 0.589 | 0.667 |
| 300 | 8,096 | 1,400 | 0.173 | 0.993 | 0.677 | 0.580 | 0.622 | 0.718 |
| 963 | 2,024 | 16,031 | 1.363 | 5.651 | 0.551 | 0.412 | 0.578 | 0.541 |
| 963 | 4,048 | 8,487 | 0.717 | 3.543 | 0.556 | 0.433 | 0.570 | 0.551 |
| 963 | 8,096 | 4,715 | 0.493 | 2.878 | 0.599 | 0.497 | 0.575 | 0.610 |

Full/8,096 and 300/2,024 have almost identical total NRMSE and window count.
The full-population run is nevertheless substantially better on the raw
thermostat-response diagnostics:

| Flexibility metric | 963 / 2,024 | 963 / 8,096 | 300 / 2,024 |
|---|---:|---:|---:|
| Event `delta Pel` NRMSE | 0.367 | 0.385 | 0.421 |
| Event-response correlation | 0.768 | 0.731 | 0.688 |
| Response-sign agreement | 0.768 | 0.757 | 0.714 |
| Upward gain absolute error [Wh/(m2 K)] | 8.19 | 9.13 | 13.17 |
| Downward gain absolute error [Wh/(m2 K)] | 4.71 | 5.59 | 9.34 |

Thus, reducing the number of buildings can look harmless in aggregate NRMSE
while materially damaging the relationship needed for flexibility analysis.

## Recommended Ablation Budget

### Architecture screening

For the expensive probabilistic contracting closed-loop model:

```text
max_profiles = 1000  # resolves to all 963 eligible HP profiles
stride = 8096
sequence_length = 960
epochs = 3 initially
fixed holdout = 20 profiles
train_eval_max_windows = 64
full_train_eval_every_epochs = 0
num_window_plots = 0
num_full_profile_plots = 0 during screening
```

This uses 3.4 times fewer windows than full/2,024. In the deterministic proxy,
mean epoch time fell from 1.36 to 0.49 minutes, while aggregate NRMSE increased
8.8% and the raw flexibility metrics remained much closer to the dense
reference than the 300-profile alternative. The measured end-to-end wall-time
speedup was 1.96x because data loading and full-profile evaluation do not scale
with stride. These are deterministic timings and must not be read as absolute
probabilistic epoch times.

For comparison, the saved
`qroom_monotone_dynamic_matrix_positive_leaky_1000` probabilistic run averaged
16.48 minutes per epoch, with a range of 15.61-17.79 minutes. It used 24
training particles on 13,090 windows and 16 evaluation particles on all 3,281
test windows each epoch. The dense deterministic study cell used one trajectory
per window and only 340 test windows. The probabilistic loss also forms
pairwise-particle energy-score and IRES terms. The deterministic study therefore
supports the relative stride/profile-quality conclusion, but an actual
probabilistic timing pilot is required to measure the final wall-time speedup.

Use 300/2,024 only as a quick implementation/debugging tier. Do not use 100
profiles to rank serious candidates: both deterministic tasks degraded to
about `0.9` total NRMSE.

### Finalist confirmation

Retrain only the top ablations with all eligible profiles and stride 2,024,
using the intended epoch count, particle counts, probabilistic KPI coverage,
and full-year plots. A candidate should advance based on a Pareto comparison of
trajectory NRMSE, temporal metrics, and raw flexibility response, not NRMSE
alone.

### Q-to-T studies

Use all 1,000 profiles. The deterministic model is cheap enough for stride
2,024. For expensive probabilistic screening, stride 4,048 is a moderate
shortcut; stride 8,096 is acceptable for coarse rejection but showed a 20%
NRMSE increase after three epochs.

## Limitations

- This is one seed and one fixed 20-profile holdout.
- Three epochs measure early ranking and cost, not convergence.
- Stride changes both sampled temporal coverage and optimizer-step count. This
  is intentional for a runtime-quality study, but it is not a pure statistical
  estimate of stride in isolation.
- Deterministic scaling is a proxy for the probabilistic model. Particle and
  proper-score costs can change absolute runtime, though the number of rollout
  windows remains the dominant multiplicative factor.
- The fixed hard-coded availability calendar is shared by all runs, so this
  study does not test availability-model uncertainty.

## Outputs

The completed study is under:

```text
output/neural_building_emulator_refactor/deterministic_scaling_study/
```

Important files are `scaling_results.csv`, `recommendations.csv`,
`deterministic_scaling_study.html`, `fixed_holdout_profile_ids.json`, and each
run's saved artifact, log, and fast-KPI report.
