# Scientific Report: Heat-Pump Turn-Off Flexibility in ODIS Simulations

## Main Conclusion

We quantify heat-pump demand flexibility in the ODIS simulation data by comparing average electrical consumption during normal operation with average consumption during a forced turn-off intervention. The central result of the analysis is the normalized turn-off flexibility index:

$$
\phi_i =
\frac{P_{i,\mathrm{baseline}} - P_{i,\mathrm{forced\ off}}}
     {P_{i,\mathrm{baseline}}},
$$

where \(P_{i,\mathrm{baseline}}\) is the mean consumption of heat pump \(i\) when no forced turn-off is active, and \(P_{i,\mathrm{forced\ off}}\) is its mean consumption during the forced turn-off state. This index measures the fraction of baseline load that can be removed by a turn-off command. Positive values therefore identify assets that reduce consumption during the intervention.

The analysis then explains this flexibility index with building and asset metadata, interprets the learned relationships with SHAP values, and maps the response across outdoor temperature and hour of day.

## Abstract

We conduct an exploratory flexibility analysis on simulated ODIS electrical load data. The study focuses on heat-pump assets and evaluates their response to a forced turn-off control signal. We first estimate a global flexibility index for each meter, then train a gradient-boosted regression model to predict this index from static metadata. We use SHAP values to identify the metadata variables that drive the model predictions. Finally, we compute conditional flexibility surfaces over temperature quantiles and hourly operating patterns, producing heatmaps that reveal when the turn-off intervention has the largest effect.

## Data

The study uses simulated ODIS data containing asset metadata, time-indexed meter consumption, intervention-state information, and meteorological variables. The analysis focuses on heat pumps, defined as:

$$
\mathcal{H} = \{i : \mathrm{asset\ type}_i = \mathrm{heat\ pump}\}.
$$

All subsequent meter-level response estimates are computed for this heat-pump population \(\mathcal{H}\). For each heat pump, the dataset provides static metadata describing the building or asset and a time series of electrical consumption.

## Metadata Diagnostic

We inspect the physical and energetic scale of the heat-pump population by relating annual consumption to floor area. For each heat pump \(i\), we define:

$$
x_i = C_i^{\mathrm{annual}},
\qquad
y_i = A_i,
$$

where \(C_i^{\mathrm{annual}}\) is annual consumption and \(A_i\) is the floor area in square meters. We color each point by a thermal transmittance-related metadata variable:

$$
\mathrm{color}_i = U_i.
$$

This diagnostic relates building size, annual energy demand, and thermal characteristics before estimating flexibility. It provides a first visual check of whether the heat-pump population spans a broad range of building scales and thermal properties.

## Global Turn-Off Response

Let \(P_i(t)\) denote the consumption of heat pump \(i\) at time \(t\), and let

$$
F(t) \in \{0,1\}
$$

denote the intervention state, where \(F(t)=0\) is baseline operation and \(F(t)=1\) is the forced turn-off state.

For each heat pump \(i\) and intervention state \(f\), we estimate mean consumption as:

$$
\bar{P}_{i,f}
= \frac{1}{|\mathcal{T}_f|}
  \sum_{t \in \mathcal{T}_f} P_i(t),
\qquad
\mathcal{T}_f = \{t : F(t)=f\}.
$$

The distributions of \(\bar{P}_{i,0}\) and \(\bar{P}_{i,1}\) describe the meter-level consumption levels under baseline and intervention states. Their difference provides the basis for estimating flexibility.

The global relative flexibility is:

$$
\phi_i
= \frac{\bar{P}_{i,0} - \bar{P}_{i,1}}{\bar{P}_{i,0}}.
$$

This sign convention makes load reduction positive. A value \(\phi_i \approx 1\) indicates that the forced turn-off removes almost all baseline load, while \(\phi_i \approx 0\) indicates little or no average response.

## Metadata-Based Flexibility Model

We model the heat-pump flexibility index as a function of numeric metadata. For each heat pump \(i \in \mathcal{H}\), the supervised learning sample is:

$$
(\mathbf{x}_i, \phi_i),
$$

where \(\mathbf{x}_i\) is the vector of metadata features and \(\phi_i\) is the global turn-off flexibility.

The data are split into training and test sets with an 80 percent training fraction:

$$
n_{\mathrm{train}} = \lfloor 0.8n \rfloor.
$$

We train a gradient-boosted tree regressor to represent the mapping:

$$
\hat{\phi}_i = f_{\theta}(\mathbf{x}_i),
$$

where \(f_{\theta}\) is the fitted ensemble model. We assess predictive performance visually on the test set by comparing \(\hat{\phi}_i\) with \(\phi_i\):

$$
\left(\phi_i, \hat{\phi}_i\right),
\qquad i \in \mathcal{D}_{\mathrm{test}}.
$$

The identity line

$$
\hat{\phi} = \phi
$$

marks perfect prediction.

## SHAP Interpretation

We interpret the trained model with SHAP. For a test sample \(i\), SHAP decomposes the prediction into a model baseline and feature-level contributions:

$$
\hat{\phi}_i =
\mathbb{E}\left[f_{\theta}(\mathbf{x})\right]
+ \sum_{j=1}^{p} s_{ij},
$$

where \(s_{ij}\) is the contribution of feature \(j\) to the prediction for sample \(i\). The SHAP summary identifies the metadata variables that most strongly explain variation in predicted flexibility and shows whether high or low feature values increase \(\hat{\phi}_i\).

## Temperature-Conditional Flexibility

We next estimate how the intervention response changes with external temperature. We define:

$$
X(t) = \mathrm{meteorological\ state}(t),
\qquad
Y_i(t) = P_i(t),
\qquad
F(t) = \mathrm{intervention\ state}(t).
$$

We remove timestamps with missing meter values or missing meteorological values. We then divide mean temperature into five quantile bins. Let \(B(t)\) denote the temperature bin assigned to timestamp \(t\). For each heat pump \(i\), temperature bin \(b\), and intervention state \(f\), we compute:

$$
\mu_{i,b,f}
= \frac{1}{|\mathcal{T}_{b,f}|}
  \sum_{t \in \mathcal{T}_{b,f}} Y_i(t),
\qquad
\mathcal{T}_{b,f} = \{t : B(t)=b,\ F(t)=f\}.
$$

The temperature-conditional relative response is:

$$
\psi_{i,b}
= \frac{\mu_{i,b,1} - \mu_{i,b,0}}{\mu_{i,b,0}}.
$$

Here we retain the signed intervention effect. Therefore, \(\psi_{i,b}<0\) indicates a reduction in consumption during forced-off periods. We visualize \(\psi_{i,b}\) as a heatmap over heat-pump meters and temperature bins, sorting meters by their response in the first temperature bin.

## Hour-and-Temperature Flexibility

We refine the conditional analysis for individual meters by adding the hour of day. For one selected heat pump \(i\), we compute the conditional mean:

$$
\mu_{i,h,b,f}
= \frac{1}{|\mathcal{T}_{h,b,f}|}
  \sum_{t \in \mathcal{T}_{h,b,f}} Y_i(t),
$$

with

$$
\mathcal{T}_{h,b,f}
= \{t : h(t)=h,\ B(t)=b,\ F(t)=f\}.
$$

The resulting response surface is:

$$
\psi_{i,h,b}
= \frac{\mu_{i,h,b,1} - \mu_{i,h,b,0}}
       {\mu_{i,h,b,0}}.
$$

Zero baseline values are excluded from the denominator to avoid undefined relative changes. The resulting hour-by-temperature heatmaps reveal how the forced turn-off response varies jointly with outdoor conditions and daily operating cycles.

## Analytical Procedure

The analysis proceeds as follows:

1. We define the heat-pump population from the ODIS simulation metadata.
2. We inspect annual consumption, floor area, and thermal characteristics.
3. We estimate baseline and forced-off mean consumption for every heat pump.
4. We compute the global flexibility index \(\phi_i\).
5. We train a gradient-boosted tree model to predict \(\phi_i\) from metadata.
6. We compare predicted and observed flexibility on the test set.
7. We use SHAP values to interpret the trained model.
8. We discretize temperature into quantile bins.
9. We estimate temperature-conditional intervention responses.
10. We visualize meter-level temperature responses as a heatmap.
11. We visualize hour-and-temperature response surfaces for selected heat pumps.

## Interpretation

The analysis formalizes a forced turn-off command as an intervention on heat-pump load. The primary empirical quantity,

$$
\phi_i =
\frac{\bar{P}_{i,0} - \bar{P}_{i,1}}
     {\bar{P}_{i,0}},
$$

is a normalized load-reduction measure. It places heat pumps with different absolute consumption levels on a common relative scale and makes the flexibility estimate comparable across buildings.

The metadata model tests whether static building and asset properties contain enough information to explain heterogeneity in \(\phi_i\). The conditional heatmaps extend the analysis from one global number per meter to operating regimes, showing how the same asset can exhibit different flexibility under different temperatures and hours.

## Methodological Scope

This study deliberately uses direct empirical averages and visual diagnostics. The following choices define the scope of the analysis:

- The forced turn-off signal is used as the intervention indicator.
- Baseline and intervention consumption are estimated with simple grouped means.
- The regression task is evaluated visually with predicted-versus-observed scatter plots.
- Two sign conventions are used: \(\phi_i\) is positive for global load reduction, while \(\psi\) is a signed conditional response where negative values indicate reduction.
- The study reports exploratory relationships rather than causal estimates adjusted for all possible confounding factors.

## Final Statement

This analysis establishes a reproducible scientific workflow for estimating and interpreting heat-pump turn-off flexibility in ODIS simulations. It measures the average load reduction available during a forced turn-off command, connects that reduction to metadata through a machine-learning model, and reveals the dependence of the response on temperature and hour of day.
