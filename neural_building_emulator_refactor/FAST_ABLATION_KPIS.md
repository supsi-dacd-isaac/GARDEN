# Fast Ablation KPIs

## Purpose

These KPIs rank model and training ablations using complete autonomous mean
trajectories. They are much cheaper than the final flexibility-calibration
analysis because they do not generate 100 scenarios and do not fit a controlled
event-study regression to every scenario.

The scorecard deliberately contains several views. Synthetic time-series
benchmarks show that point error, marginal fidelity, temporal dependence, and
downstream behavior can rank the same generators differently. A single
aggregate number would hide exactly the architectural failures of interest
here.

## 1. Normalized Trajectory Error

For channel `j`, let `s_j` be its standard deviation fitted on the training
targets. The primary scalar error is

```text
total_nrmse = sqrt(mean_t,j [((y_hat[t,j] - y[t,j]) / s_j)^2]).
```

This gives `Tin`, `Qroom`, and `Pel` equal weight after training-set
normalization. It is more stable than dividing by each profile's standard
deviation: winter temperature or inactive-power windows can have almost zero
variance. Channel RMSE, MAE, bias, and profile-standard-deviation NRMSE are
also retained as diagnostics.

Lower is better.

## 2. Marginal Fidelity

For every channel, compute the empirical one-dimensional Wasserstein distance
between the complete simulated and emulated samples, divided by `s_j`:

```text
W1_norm_j = W1({y[t,j]}, {y_hat[t,j]}) / s_j.
```

This detects a wrong operating range, marginal spread, or persistent offset.
It ignores timing by design, so it complements rather than replaces RMSE.

Lower is better.

## 3. Temporal Fidelity

### Autocorrelation error

At default lags

```text
[1, 2, 4, 8, 12, 24, 48, 96]
```

corresponding to 15 minutes through 24 hours, compute

```text
ACF_MAE_j = mean_lag |ACF_hat_j(lag) - ACF_j(lag)|.
```

This measures memory and persistence. If one trace is constant and the other
is not, the distance is defined as one; two constant traces have distance zero.

### Increment-spectrum distance

First difference each signal, estimate its Welch power spectrum, normalize the
spectrum to sum to one, and compute the square root of the Jensen-Shannon
divergence normalized by `log(2)`:

```text
spectral_JS_j = sqrt(JS(PSD(diff(y_hat_j)), PSD(diff(y_j))) / log(2)).
```

The result lies in `[0, 1]`. Differencing prevents annual level and seasonal
drift from masking cycling, abrupt changes, and high-frequency artifacts. The
evaluator also records the simulated increment energy in four interpretable
bands: faster than 1 hour, 1-6 hours, 6-36 hours, and slower than 36 hours.

Both temporal distances are lower when better. They should remain separate:
ACF emphasizes time-domain memory, while the spectrum exposes frequency
allocation.

## 4. Raw Three-Hour Flexibility Response

At every clean setpoint discontinuity `k`, define

```text
delta_Tset[k] = Tset[t_k] - Tset[t_k - 1]
P_pre[k]      = mean(Pel[t_k - N : t_k])
P_post[k]     = mean(Pel[t_k : t_k + N])
delta_P[k]    = P_post[k] - P_pre[k]
```

where `N = H / dt`, with `H=3 h` by default. A clean event has no additional
setpoint jump inside either window. Events smaller than `0.05 C` are ignored.

The most direct emulation score is

```text
event_delta_P_nrmse = RMSE(delta_P_hat - delta_P) / s_Pel.
```

The event correlation, response-sign agreement, and MAE are also reported.
Unlike the final KPI, this requires no controls or regression.

For each direction, the raw power and energy gains are

```text
gain_P_direction = mean_k(delta_P[k] / delta_Tset[k])
gain_E_direction = H * gain_P_direction.
```

Their units are `W/(m2 K)` and `Wh/(m2 K)`. Since both numerator and
denominator are negative for an expected downward response, a physically
correct downward gain is positive.

The ordinary percentage change is undefined or explosive when `P_pre` is zero.
The optional relative descriptor is therefore the symmetric change

```text
2 * (P_post - P_pre) / (|P_post| + |P_pre| + 0.1 W/m2),
```

which is bounded and should not be used as the primary physical KPI.

## What the Initial Check Shows

The first check used 20 identical held-out profiles from two saved contracting
probabilistic artifacts and two evaluation particles. A complete scorecard took
about 20 seconds per artifact after loading and compilation.

For the positive-leaky artifact:

| Metric | Mean |
|---|---:|
| total NRMSE | 0.689 |
| temperature ACF MAE | 0.031 |
| temperature increment spectral JS | 0.480 |
| Pel increment spectral JS | 0.395 |
| raw event `delta Pel` NRMSE | 0.431 |
| raw event `delta Pel` correlation | 0.817 |

Across profiles, total NRMSE had Spearman correlation `0.73` with event-response
NRMSE, but only `0.21` with temperature spectral mismatch and `-0.06` with
temperature ACF error. The temporal scores therefore identify behavior not
captured by point error. Simulated upward and downward raw flexibility gains
were highly correlated (`0.90`), so both should be shown for asymmetry but not
counted as independent evidence in a composite score.

The sample is too small to rank architectures conclusively. Its purpose is to
confirm that the metrics are numerically stable and nonredundant.

## Recommended Ablation Protocol

1. Keep dataset, ordered profile split, scaler fit, seed set, and evaluation
   particle count identical. The comparison CLI rejects different split IDs.
2. Use 100 buildings only for implementation smoke tests. The deterministic
   scaling study found that this tier was too inaccurate to rank candidates.
3. For serious closed-loop screening, retain all eligible buildings and use
   stride 8,096. Use 300 buildings/stride 2,024 only as an intermediate debug
   tier; it degraded the raw flexibility relationship despite similar total
   NRMSE. See [DETERMINISTIC_SCALING_STUDY.md](DETERMINISTIC_SCALING_STUDY.md).
4. Score a fixed 20-50 profile holdout with four mean-evaluation particles.
5. Compare median and p90 as well as mean. A model with a good mean and bad p90
   is still unreliable.
6. Do not collapse everything immediately. Select on a Pareto view of:
   `total_nrmse`, temperature temporal mismatch, Pel temporal mismatch, and raw
   event-response NRMSE.
7. Retrain only the Pareto candidates with stride 2,024 and the final epoch and
   particle counts.
8. Run the expensive 100-scenario event-study coverage analysis only on those
   finalists. The fast scorecard evaluates mean fidelity, not probabilistic
   sharpness or calibration.

## Literature Basis

- [TimeGAN (NeurIPS 2019)](https://proceedings.neurips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html)
  evaluates both sample similarity and predictive utility, emphasizing that
  temporal dynamics must be assessed explicitly.
- [COT-GAN (NeurIPS 2020)](https://papers.nips.cc/paper/2020/file/641d77dd5271fca28764612a028d9c8e-Paper.pdf)
  directly compares auto- and cross-correlation structures of real and
  generated sequences.
- [TSGBench](https://arxiv.org/abs/2309.03755) uses a suite of complementary
  measures and studies how model rankings change across them.
- [AEC-GAN (AAAI 2023)](https://ojs.aaai.org/index.php/AAAI/article/download/26208/25980)
  evaluates long autoregressive generation with ACF, marginal moments, learned
  representation distance, and downstream forecasting, and documents how
  long-horizon distribution shift can amplify generation errors.

Learned discriminative, predictive, or representation metrics are useful for a
general synthetic-data benchmark. They are not the default here because they
would require fitting auxiliary models, are harder to interpret physically,
and are too expensive for the inner ablation loop.
