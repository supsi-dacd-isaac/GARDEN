# Two-temperature structured experiment

The optional `prob_tank_temperature_state` model replaces the independent
hidden buffer-energy recurrence with two effective temperature states, SH and
DHW. The public predictions remain Tin, Qroom and Pel_SH. Both temperatures are
available as auxiliary predictions for training and annual diagnostic plots.

## Generic Assumptions

- Effective capacities are `1163 * volume_m3 / heated_area_m2` Wh/(m2 K),
  multiplied by a learned metadata/annual-latent factor bounded to [0.25, 4].
  These corrections represent unresolved stratification and effective volume;
  they are learned parameters, not independent uniform random multipliers.
- SH heat delivery is demand-limited, energy-limited and gated monotonically by
  the predicted SH-to-room temperature difference. The existing HP controller
  additionally receives both predicted tank temperatures.
- SH temperature evolves from predicted COP * Pel_SH, delivered heat and
  standing losses. A learned [0.25, 4] ratio relates the modeled zone's W/m2
  heat to the whole-building storage boundary. It is not observed at inference.
- DHW temperature evolves from known withdrawal, nominal recovery capacity,
  learned recovery demand and standing losses. Recovery is unavailable during
  force-off and during the sampled SH-active fraction. This is a shared-source
  approximation, not a reproduction of EnergyPlus's mode or node controller.
- Known seasonal availability informs SH initialization. Other initial-state
  uncertainty comes from metadata and the annual latent, never measured tank
  temperatures. There are two additional independent bounded-scale Gaussian
  process innovations, drawn per particle and timestep.
- States have numerical safety bounds of 1 to 90 C. A successful model must
  not achieve its fit by persistently saturating these bounds.

The exported SH signal is the mean tank-node temperature averaged over a
reporting interval, not a tank endpoint or outlet temperature. The observation
model uses the midpoint of consecutive effective states. This is an explicit
coarse approximation; the two states do not represent eight stratified nodes.

## Labels and Loss

`sh_tank_measured_temperature_c` on `technical_room` maps to canonical
`sh_tank_mean_temperature_c`. DHW uses `dhw_tank_mean_temperature_c` from the
same entity. Both are aligned with the existing interval targets after the
first row. They occupy columns 3:5 of the separate plant-label array and never
enter the input array. Existing plant labels in columns 0:3 are unused by this
experiment. Missing temperatures fail loudly.

The new loss is the sum of the two marginal temperature CRPS terms, normalized
by a fixed 10 C scale and scaled like the existing path loss by sqrt(length).
The first four hours of each training window are excluded from this auxiliary
loss (capped to one quarter of a very short window). The original three-target
losses, training hard-switch scheme and checkpoint criterion remain unchanged.

## Controlled Comparison

`python -m neural_building_emulator_refactor.tank_temperature_experiment --stage prepare`

The v9-final campaign's fold membership is preserved. Seed 13 selects 140
training, 20 validation, 20 reserved calibration and 20 test buildings. Both
models train from scratch on the two excitation policies for 15 epochs with
the current selected 0.5-hour model's other settings. Both receive DHW volume
and recovery metadata, so this is an architecture/supervision comparison,
not a comparison of different available metadata. The baseline's legacy energy
cap is retained; the candidate replaces that buffer recurrence.

Before a full run, `--stage smoke --variant two_tanks` trains a tiny model,
reloads the saved checkpoint, and checks autonomous full-year inference.
`--stage supervise` runs both training jobs, native annual evaluation on all
four policies, auxiliary tank diagnostics, and a paired report sequentially.
Scoring uses 100 full-year particles per test building, native regression and
direct-F KPIs, coverage, sharpness, PIT and building-bootstrap CRPS contrasts.
No RNS, reweighting or other ex-post recalibration is fitted. This is a
single-seed screen, not evidence of robustness across training seeds.

Sources, dataset completion records, splits and checkpoints are fingerprinted.
Training artifacts and smoke tests are separate; no historical checkpoint is
overwritten. Final plots include the original three signals and both tank
temperatures with native 95% ensemble bands.

## Full 1000-building validation-gated run

Prepare a fresh output directory with `--max-profiles 1000 --epochs 10
--validation-epoch-gate`. This retains all original 701/100/99/100 train,
validation, reserved-calibration and test buildings. Neither the tariff
policies nor the test/calibration folds affect stopping.

Both arms have a ten-epoch initial budget and a fifteen-epoch ceiling. At
epochs 8 and 10 the trainer pauses while separate workers score continuous
annual rollouts on all 100 validation buildings under both excitation
policies, using 50 particles and fixed matching random seeds. The epoch-10
decision averages relative CRPS changes within the shared three-target trace
group and the four policy/KPI-family group, then gives those two groups equal
weight. Continue to 15 only if that combined improvement is at least 1%, and
neither group deteriorates by more than 1%. This is a fixed budget heuristic,
not a significance test. Scores and decisions are saved with the run.

The optimizer, random state and model stay alive during validation. Window
rotation follows the same fixed fifteen-epoch schedule in both arms, including
when stopped at 10. Checkpoint selection still minimizes original-target
validation NRMSE across completed epochs; the extra tank loss cannot select a
checkpoint on the baseline's behalf. Test scoring uses 100 particles and all
four policies. The source snapshot and request record the initial budget and
conditional maximum separately.

The supervisor updates `status.json` atomically every 30 seconds with worker
status, log activity, completed epoch, validation error and nonfinite/skipped
batch counts. Nested annual-validation progress is included. Worker failure
stops the pipeline with a failed status; it does not silently launch the next
stage. `epoch_decision.json` records each arm's continuation decision.
