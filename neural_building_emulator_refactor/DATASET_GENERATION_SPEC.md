# Open Loop Dataset Generation Specification

## Handoff Task

Implement a reproducible EnergyPlus dataset campaign for training and testing
building emulators under different **precomputed, open-loop control policies**.
Generate four matched annual datasets: two for training and two for policy
transfer. Preserve compatibility with GARDEN's existing dataset reader and KPI
estimators. First implement schedule generation, tests, manifests, and a pilot;
obtain approval before launching the full simulation campaign.

The goal is informative thermal/HP excitation and a fair test of policy
generalization, not a dataset engineered to make LightGBM lose. Do not retrain
emulators, change their architecture, recalibrate predictions, overwrite existing
datasets, or reimplement the simulator as part of this task.

**Hard constraint:** the external policy must not read simulated indoor
temperature, buffer state, HP operating state, delivered heat, or comfort.
Generate the entire command schedule before launching EnergyPlus, using only
calendar, a frozen tariff series, static configuration, and seeded randomness.
The simulator's existing internal thermostat, plant, and ventilation controls
remain unchanged; their ordinary dependence on simulated state is allowed.

Numerical schedule settings below are proposed pilot defaults, not established
optimal values. Freeze any pilot-approved revisions before held-out evaluation.

## Repository Contracts

Start by reading these files in the GARDEN repository:

- `neural_building_emulator/dataset_adapter.py`: entity-oriented data contract.
- `neural_building_emulator/data.py`: alignment, area normalization, DHW masking.
- `neural_building_emulator/columns.py`: canonical names and input schema.
- `neural_building_emulator_refactor/README.md`: current evaluation protocol.
- `neural_building_emulator/flexibility_event_study_kpis.py`: setpoint KPIs.
- `neural_building_emulator_refactor/intervention_flexibility.py`: force-off KPIs.
- `neural_building_emulator_refactor/policy_kpi_identifiability.py`: residualized
  directional-rank check. Reuse its function; its CLI is tied to an existing run.

Use these existing data folders as schema/configuration references, not output
destinations:

```text
data/control=setpoint_identification_dhw_fixed/
data/control=force_off/
data/control=setpoint/
```

Existing metadata identifies control providers `coolflex.heating_gbn` and
`coolflex.odis`. Locate the actual producer and its supported control-file
interface before implementing the schedules. Do not assume the older scripts in
`building_simulation/` produced these exports. If the producer is in another
repository, document that dependency and the exact interface required; do not
invent a working simulation CLI or silently substitute another plant model.

The current benchmark root is:

```text
output/neural_building_emulator_refactor/current_best_cap0p5_lstm_markov_1000_e15/
```

Its `split_manifest.json` contains **606 eligible buildings**, despite the 1000
limit in the run name: 423 training, 61 validation, 61 calibration, and 61 test.
Its `comparison_request.json` is the reference for evaluation settings.

The previous transfer audit, under that root's `tariff_policy_transfer/`, found:

- Physical/DHW configuration differences between source and target exports.
- A two-level tariff setpoint policy with many events but no independently
  identifiable up/down coefficients after the full nuisance controls.
- `kpi_identifiability_summary.csv` reports residualized directional rank 2 for
  source and 0 for tariff, in both calibration and test folds.

These are concrete failure modes the new campaign must prevent. More events
alone do not resolve rank deficiency.

### What Is Available From Git

The specification, dataset reader, KPI estimators, and reusable
`directional_identifiability(design)` function are source-code deliverables.
Importing that function requires NumPy, not trained weights or benchmark outputs.
The module's historical command-line audit additionally requires compatible
trained artifacts and the local source/target data; it is not a new-dataset CLI.

The `data/` and `output/` folders are intentionally ignored by Git. The historical
split manifest, simulation metadata, tariff files, checkpoints, and audit CSVs
must be supplied separately through an approved shared location to reproduce
that exact cohort/audit. Do not assume a fresh clone contains them, reconstruct
the historical split from counts alone, or fabricate the missing producer/data.
The self-contained requirements and defaults below remain usable to implement
the new schedule generator before those historical artifacts are available.

## Four Dataset Definitions

Proposed new output roots; confirm producer naming support without overwriting
the existing roots:

| ID | Proposed root under `data/` | Role | External controls |
|---|---|---|---|
| A | `control=setpoint_excitation_v2` | Training | Random multilevel Tset; force-off always 0 |
| B | `control=force_off_excitation_v2` | Training | Random HP force-off; nominal Tset |
| C | `control=tariff_setpoint_v2` | Transfer | Tariff-biased multilevel Tset with exploration; force-off always 0 |
| D | `control=tariff_force_off_v2` | Transfer | Tariff-biased HP force-off with exploration; nominal Tset |

Use the same building cohort in all four. The A/C pair tests transfer of
setpoint policy; B/D tests transfer of force-off policy. Training on A+B is
permitted only for training-fold buildings. Target-policy data from C/D must
not be used to fit the primary frozen-model benchmark.

Keep D's setpoint nominal in the primary campaign so the comparison isolates
force-off timing. A scripted preheat -> shutdown -> recovery policy is an
optional, separately versioned compositional extension, disabled by default.
It needs no state feedback, but must not silently replace D or be described as
a pure force-off-policy comparison.

## Shared Physical and Temporal Configuration

1. Select one approved physical configuration per building, preferably the
   existing fixed-DHW source configuration. Size the building/HP/buffers once
   under a common design procedure, then reuse the resolved design in A-D.
   Do not resize equipment from each policy's simulated demand or peak.
2. Freeze geometry, envelope, thermal mass, infiltration, ventilation rules,
   cooling settings, emission system, HP model/capacity/COP curves, SH tank,
   DHW tank/delivery/storage setpoints, hysteresis, backup heater, and plant
   topology. A missing metadata field is not proof of equal resolved physics.
3. Reuse identical weather, solar forcing, occupancy, non-people gains, DHW
   withdrawal inputs, seasonal availability, and all non-policy random seeds.
   Realized ventilation, DHW charging, HP modes, and temperatures may differ
   because they are endogenous outputs; do not require these to match.
4. Use the existing non-leap 2023 weather/calendar for the first campaign and
   15-minute intervals. Raw annual schedules contain 35,040 timestamps.
   Record the producer's timezone, interval-start/end convention and DST
   handling. Never introduce duplicate/missing intervals by naive timezone
   conversion. Model alignment may yield 35,039 samples; that is not a missing
   raw simulation interval.
5. Use identical simulator warm-up settings and nominal controls during
   warm-up, followed by a common 48-hour nominal prefix in the recorded year.
   Exclude prefix interventions from KPI analysis and record the mask. Do not
   reset indoor/buffer states between months or intervention blocks.
6. Respect the producer's explicit seasonal SH-availability schedule. Do not
   invent a second temperature-dependent availability rule. Outside that
   schedule use nominal Tset and no external shutdown, while retaining all
   annual outputs, DHW operation, and summer data.
7. Define force-off semantics from the producer: compressor vs SH branch,
   treatment of DHW priority, backup heat, delays, and requested vs effective
   commands. Freeze them across B/D. A compressor shutdown must not be
   implemented by zeroing zone heat, ventilation, or stored buffer energy.

A-D must have identical canonical physical/exogenous configuration hashes per
building. Maintain an explicit allowlist for policy-specific fields. Runtime
IDs and outcome diagnostics can differ, but DHW, sizing, or resolved plant
parameters must never be hidden by that allowlist.

## Cohort and Split Rules

- Default to the exact existing 606-building manifest for the matched rerun.
  All policy copies and any repeated seeds for a building belong to one fold.
- If 1000 genuinely eligible buildings are required, expand the cohort before
  simulation, preserve existing fold assignments, assign new IDs deterministically
  near 70/10/10/10 proportions, and write a new versioned manifest. Report the
  actual eligible count; never relabel 606 as 1000.
- Select pilot buildings from the training fold only. Cover available ranges of
  floor area, thermal mass, HP family/sizing, SH volume per heated area, and DHW
  demand. Do not select buildings based on which model performs worst.
- The historical test fold has already been inspected. Label it a historical
  holdout, not a pristine confirmatory test. Reserve previously unseen buildings
  if a new confirmatory result is needed.
- Record failures and retries. Use a common complete four-policy cohort for
  paired comparisons and publish all exclusions. Do not silently choose a
  different intersection per model, or drop buildings for uncomfortable outcomes
  or difficult-to-predict traces.

## Common Schedule Generator

Produce the complete schedule and its provenance before simulation. Use a
documented stable seed derivation from study seed, EGID, policy family, replicate,
and named random stream; do not use Python's process-randomized `hash()`.
Keep exogenous-realization seeds separate from policy seeds. Reuse proposal
streams across A/C and B/D where practical, with separate selection streams.

Export at least timestamp, Tset command, requested force-off, nominal Tset,
SH availability, policy/block/event IDs, initialization mask, and exploration
flag. Save tariff values and units for C/D. Export planned duration, recovery
constraint, admissible choices, and selection probabilities where available;
distinguish proposal probabilities from final probabilities after constraints.

Bounds and recovery restrictions depend only on configuration and the planned
schedule. Never shorten a shutdown because a simulated room becomes cold, wait
for a simulated buffer to refill, or reject a schedule based on target KPIs.

### A Random Setpoint Excitation

Let `T0_b` be the building's nominal heating setpoint and use piecewise-constant
`Tset_b(t) = T0_b + offset_b(t)` during the available heating season.

| Parameter | Initial default |
|---|---|
| Offset levels, degC | `[-2, -1, -0.5, 0, 0.5, 1, 2]` |
| Level proposal weights | `[0.10, 0.15, 0.15, 0.20, 0.15, 0.15, 0.10]` |
| Block durations, hours | `[0.5, 1, 2, 3, 6, 12, 24, 48]` |
| Duration probabilities | `[0.05, 0.05, 0.10, 0.15, 0.25, 0.20, 0.15, 0.05]` |
| Maximum setpoint jump | `4 degC` |
| External force-off | Always `0` |

At each block boundary draw the next admissible level, excluding the current
level and renormalizing weights. Dwell time and level draws must not depend on
simulator outputs. Retain zero-offset blocks and long dwells, not just frequent
pulses. Different magnitudes of upward/downward changes must occur from shared
pre-setpoint levels, rather than forcing alternating fixed +/-1-degree jumps.

Check absolute heating bounds and heating/cooling deadband against the approved
configuration before generation. If levels are infeasible, document the frozen
per-building admissible set and use the same set in C. Do not silently clip
commands, alter cooling settings, or remove enough levels to destroy rank.
An offset bound of +/-2 permits a 4-degree jump; these are different constraints.

The previous source used a wider offset range and 3-48-hour dwells. The defaults
above deliberately add short excitation while retaining long memory probes;
they are a new protocol, not an exact reproduction of the previous schedules.

### B Random Force Off Excitation

Hold `Tset = T0_b`. Randomize the requested HP force-off schedule independently
of tariff, clock-time preferences, and simulated state, subject to these bounds:

| Parameter | Initial default |
|---|---|
| Probability of proposing an event on an eligible day | `0.75` |
| Maximum newly starting events | `1` per calendar day |
| Shutdown durations, hours | `[0.25, 0.5, 1, 2, 4, 6]` |
| Duration probabilities | `[0.10, 0.15, 0.20, 0.20, 0.30, 0.05]` |
| Minimum unrestricted interval after release | `max(6 h, 2 * preceding shutdown duration)` |
| Maximum shutdown time | `6 h` in any rolling 24-hour period |
| Start time | Uniform over feasible 15-minute slots, including daytime/nighttime |

Sample the duration before selecting a start. Place only complete events inside
SH availability and outside initialization, with complete pre/post observation
windows. Respect recovery and rolling limits across midnight. Record infeasible
proposals instead of silently changing their durations. Preserve untreated days
and long unrestricted windows for the regression's reference observations.

Longer shutdowns are intended to expose buffer depletion and slower response;
they do not guarantee depletion in every building. Retain short shutdowns to
probe fast dynamics. This design broadens durations rather than simply keeping
the HP off more often. Quantify the realized event mix after scheduling constraints.

### C Tariff Setpoint Transfer

Use A's admissible levels, dwell-duration proposals, bounds, and nominal prefix.
Change the association between prices and levels, not the plant configuration.
Use multilevel piecewise-constant blocks, not binary tariff switching or a new
small setpoint jump every 15 minutes.

One implementable default is a mixture at each block boundary:

```text
z(t) = 2 * within-day percentile rank of tariff(t) - 1
z_block = mean z(t) over the proposed block
v_j = admissible offset_j / 2 degC
p_tariff(j) proportional to base_weight_j * exp(-2 * z_block * v_j)
p(j) = 0.8 * p_tariff(j) + 0.2 * p_base(j)
```

Normalize both distributions over admissible levels excluding the current level.
Define percentile rank as `(one_based_midrank - 0.5) / number_of_daily_slots`,
giving `z=0` on flat-price days. Low prices favor positive offsets;
high prices favor setbacks. Record which mixture component generated the choice.
The exploration component and multiple amplitudes support identification but
do not replace the actual rank/conditioning checks below.
Whole-day averages of this within-day signal can be zero, so long blocks may
deliberately reduce to the random policy. Check that shorter blocks produce a
meaningful realized tariff association in the pilot; do not assume it from the
policy name. Any change to price normalization must be versioned and frozen.

Archive one specified tariff file, its units, calendar mapping, and checksum.
Expand hourly values without inventing subhour price variation. This is offline
schedule generation using the archived tariff over the whole proposed block,
not a claim of a deployable day-ahead policy when a block exceeds that forecast
horizon. Missing tariff data must fail validation or use a documented, frozen
imputation rule. Do not fabricate a tariff if its source is unavailable.

### D Tariff Force Off Transfer

Use B's nominal Tset, event-day probability, duration proposals, recovery limits,
daily/rolling limits, and requested-command semantics. For a proposed duration,
enumerate feasible start slots and change only their selection distribution:

```text
score(start) = mean z(t) during the proposed shutdown
p_tariff(start) proportional to exp(2 * score(start))
p(start) = 0.8 * p_tariff(start) + 0.2 * uniform(feasible starts)
```

This favors shutdown during expensive periods while retaining untreated and
treated observations across price/clock regimes. Retain variation in duration;
do not turn all high-price hours into one deterministic binary schedule. Record
realized price association, event density, duration mix, and the fraction of
proposals rejected by temporal constraints for both B and D.

## KPI Identifiability and Event Support

Keep the current reference horizons `H = [0.5, 1, 2, 3] h`, timestep `0.25 h`,
setpoint event threshold `0.05 degC`, and legacy minimum event count `20`.
Use the exact existing estimators for the comparable benchmark. Additional
diagnostics must not silently redefine the benchmark KPI.

### Setpoint KPIs for A and C

The reference regression uses the power pre/post response, directional setpoint
jumps, and `controls="full"`, including pre-power, pre-temperature, pre-setpoint,
previous jump, and disturbance/calendar controls. Audit its actual design, not
just the commanded schedule or total matrix rank:

```text
D = [max(delta_Tset, 0), min(delta_Tset, 0)]
N = [intercept, all nuisance columns from the reference estimator]
R = D - N @ lstsq(N, D)
require rank(R) == 2
```

Use the existing relative tolerance `1e-8` for the algebraic rank check. Also
report both singular values relative to `||D||_2`, their ratio, residualized
column fractions, and event counts by direction/horizon. Set a practical weak-
identification warning threshold from the training-only pilot and freeze it;
a barely nonzero singular value is not strong identification.

Offline schedule checks can assess excitation and calendar/tariff confounding.
The full check requires simulated pre-state and other realized controls, so run
it after the pilot and again for every generated building. Do not claim that
full rank proves causal identification; some current controls include realized
post-event disturbances, and direct-F is a descriptive policy-conditional KPI.

Aim for at least 50 usable events of each sign per horizon in the heating
season, while reporting the unchanged legacy score and its event selection.
Also count events with no other Tset jump in the H-hour pre/post windows. The
current setpoint estimator does not enforce this isolation; report an isolated
event sensitivity analysis separately, not as a replacement without versioning.

### Force Off KPIs for B and D

Do not apply the two-direction Tset-rank requirement to constant-Tset data.
The force-off implementation uses stable onset windows: unrestricted for H
before the onset and continuously force-off for H afterward. Consequently a
1-hour shutdown cannot contribute to the existing 3-hour force-off KPI.

The direct score is `H * mean(P_pre - P_post)` for eligible binary onsets. The
regression compares onsets with untreated reference windows and uses one binary
treatment column, not two directional Tset columns. Require that treatment
retain nonzero residual variation after the exact nuisance controls. Report
treated/reference counts, residual treatment norm, conditioning, and overlap in
price/time/weather/pre-state; rank alone is not adequate overlap.

Require at least the existing 20 eligible onsets at every scored horizon; target
40 or more, plus enough untreated windows for the existing reference sampler
(which requests up to four per onset). The 4/6-hour duration tail is essential
for the 3-hour cell. Flag insufficient support instead of manufacturing a score.
Report release/rebound responses separately. Do not confuse electrical shed
with loss of room heat or temperature: stored heat can continue after shutdown.

For all datasets, report unavailable/invalid KPI cells explicitly, including
zero-demand cases. Never select events by a favorable observed temperature drop
or discard an otherwise valid event because the HP happened to be inactive.
If pilot support fails, revise the common generator before freezing the campaign;
do not search held-out seeds until target outcomes look favorable.

## Export Contract and Provenance

Preserve the existing per-building layout:

```text
data/control=<new-policy>/egid=<ID>/
  simulation_metadata.json
  timeseries.parquet
  complete.json
```

The parquet is entity-oriented: one row per timestamp/entity, not one total row
per timestamp. Preserve `datetime`, `zone`, the modeled zone `fl0_thz0`, weather
entity `building_total`, and HP entity `technical_room`. Preserve other simulated
zones and the established mappings. Inapplicable entity-channel NaNs are valid;
missing required values on the appropriate entity are not.

Required channels include:

| Channel | Native export unit / purpose |
|---|---|
| `Zone Air Temperature` | degC; zone end-of-interval state |
| `Zone Thermostat Heating Setpoint Temperature` | degC; applied setpoint |
| `zone_thermal_heating_power` | W; zone interval heating power |
| `heat_pump_electric_power` | W; technical-room HP electricity |
| `hp_mode_is_dhw`, `hp_effective_mode` | Existing mode/masking semantics |
| `hp_force_off_requested`, `hp_force_off_effective` | Explicit requested/effective commands |
| `input_non_people_internal_gain_net_kw` | kW |
| `input_occupants_present` | persons |
| `input_dhw_mixed_water_l` | L per 15-minute interval |
| `space_heating_available` | Binary seasonal availability |
| `Environment, Site Outdoor Air Drybulb Temperature` | degC |
| `Environment, Site Global Horizontal Solar Radiation Rate per Area` | W/m2 |
| `Zone Ventilation Standard Density Volume Flow Rate` | m3/s |

Use the producer's declared schema and `timeseriesColumnUnits` as the authority;
resolve any naming/semantic discrepancy explicitly. Keep raw W/kW values in
native units, not already normalized. The current adapter normalizes zone heat
by zone `floor_area` and whole-building HP power by `floor_area * totalFloors`,
with DHW masking; validate those area meanings for every exported building.
Preserve `gardenCompatibility.values`, units, and required metadata including
SH/DHW volumes, HP/SH capacities, COP, model identifier, area and floor count.

Archive existing SH/DHW storage-charge/delivery channels, mode signals, and
buffer temperatures/energy when readily available from the producer. These are
diagnostic outputs, not mandatory emulator inference inputs. Do not add new
simulator internals merely to satisfy this specification.

Classify every channel as an exogenous input/command, prediction target, or
diagnostic. Requested force-off is known before rollout; effective force-off,
HP mode, buffer state, and realized ventilation may depend on simulator state.
Do not automatically expose those outputs as oracle inputs at inference. Mode
channels may still be used for the established construction of SH-only targets.
Document any permitted initial observation separately from annual future inputs.

At campaign level export a frozen configuration, split manifest, schedule files,
event table, exclusion/failure table, and provenance manifest. Include source
repository commits/dirty-diff hashes, EnergyPlus version, weather/tariff hashes,
seed derivation and seeds, resolved per-building configuration hashes, control
hashes, units, timestamp semantics, warm-up/prefix settings, and requested vs
completed profile counts. Completion markers must be written only after output
validation. Resume only when configuration/input hashes agree; never reuse an
old cache merely because EGID and a broad control label match.

## Validation and Pilot Gates

1. **Schedule-only checks for the full planned cohort:** deterministic reruns,
   distinct building streams, exact annual grids, admissible levels, binary
   commands, dwell/duration distributions, recovery/rolling limits, heating
   availability, tariff association, event-window support, and output hashes.
   Instrument the generator or test its interface to establish that it cannot
   receive simulated state. These checks need no EnergyPlus execution.
2. **Producer contract smoke test:** two training buildings across A-D, short
   runs if supported, to verify command injection, entity/unit mapping, requested
   vs effective off behavior, and time alignment. Short traces are not annual
   KPI/calibration evidence.
3. **Annual pilot:** 12 training buildings across A-D, covering the metadata
   ranges above. Check exact physical/exogenous matching, load through the
   existing GARDEN reader, and compute event/rank/conditioning diagnostics.
   Validate the force-off horizons with the actual scorer, not onset counts
   alone. Confirm nominal-prefix reproducibility to numerical tolerance.
4. **Pilot report and approval:** show realized schedules, annual Tin/Qroom/Pel
   plots, representative onset/release zooms, temperature-drop distributions,
   annual energies, HP on-fraction, DHW fraction, ventilation-active fraction,
   comfort degree-hours relative to an explicitly specified reference, all
   event-support failures, and simulator warnings. Comfort is an outcome, not
   feedback or an exclusion criterion. Fix numerical/export errors, not genuine
   inconvenient trajectories. Report planned runtime/storage from the pilot.
5. **Full campaign after approval:** freeze protocol/seeds, run resumably with
   bounded parallelism, validate outputs, and publish the same audit tables for
   all buildings. Quarantine true simulation/export failures with reasons.
   Retain weakly identified cases and mark their affected KPIs invalid; do not
   silently replace buildings or tune policy settings on test outputs.

Unit tests must include a deliberately rank-deficient binary Tset schedule, a
multilevel design with independent directional columns, horizon eligibility for
short vs long force-off events, no-event/flat-tariff cases, midnight/year-edge
constraints, missing-channel failures, paired-NaN-safe entity comparisons, and
the existing reader's one-interval alignment. Use lightweight synthetic fixtures
for tests; full annual simulation belongs to the pilot, not the unit suite.

## Downstream Comparison Requirements

The generation deliverable should enable, but need not execute, this protocol:

- Fit structured, vanilla LSTM, Markov-emission LSTM, and LightGBM using the same
  A/B training buildings; use validation only for model selection and the source
  calibration fold only for source-fitted recalibration. No target-policy test
  labels may affect training, normalization, feature engineering, or calibration.
- Evaluate native continuous annual rollouts on held-out A/B and C/D. Keep
  original KPI estimators, horizons, units, particle budget, and split membership
  matched across models. Score C's setpoint KPIs and D's force-off KPIs separately.
- Report raw and source-fitted RNS/RNS-ECC results separately. Any calibration
  fitted on C/D calibration buildings is a separate target-adaptation experiment,
  not zero-shot transfer. RNS/ECC KPI calibration does not produce calibrated
  physical time traces; label annual bands accordingly.
- Keep the historical metadata-only LightGBM baseline, but also allow a fairly
  trained policy-aware baseline with schedule/tariff/weather summaries available
  without simulation. Build features without target outcomes. A frozen
  metadata-only predictor cannot react to different schedules for the same
  building; that limitation alone is not evidence of superior dynamics learning.
- Report policy-marginal and sequence differences. Prefer transfer within the
  training amplitude/duration support, with changed tariff association and
  temporal ordering, rather than an artificially extreme target regime. No
  dataset design can guarantee an emulator will outperform LightGBM.
- Evaluate per-building KPI CRPS, native PIT/coverage/sharpness, joint dependence,
  energy and comfort errors, and paired building-bootstrap differences. Keep
  policies/replicates of one building together when resampling. These diagnostics
  assess calibration across buildings; one deterministic simulator realization
  per building does not establish its full conditional uncertainty distribution.

## Required Deliverables

- Optional/configurable schedule generators for A-D, their unit tests, and a
  documented producer adapter using the actual supported interface.
- Frozen policy/physics configurations, cohort/split manifest, complete command
  schedules and hashes, and a dry-run validation report.
- Matched annual pilot outputs, compatibility/identifiability audits, trace
  plots, failure log, runtime/storage estimate, and a clear full-run go/no-go.
- After approval, the four annual dataset roots, reproducible launch/resume
  instructions, shared-cohort manifest, and final acceptance report.

In the handoff response, state what is implemented, what was actually simulated,
which gates passed, which cells/configurations remain unsupported, and any
missing external producer or tariff dependency. Never claim completion of a
1000-building campaign from a request limit or a partially populated directory.
