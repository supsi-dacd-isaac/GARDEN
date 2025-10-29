import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from lightgbm import LGBMRegressor

data = pd.read_pickle('data/ODIS/sim_results_models_odis_long_raw_data_power_with_uncontrolled.pk')


meta_hp = data['meta'].loc[data['meta']['type'].isin(['hp'])]

# scatter annual consumption vs meters**2 colored by U
meta_small = meta_hp.sample(n=1000, random_state=0) if len(meta_hp) > 1000 else meta_hp
sns.scatterplot(data=meta_small, x='annual_consumption', y='m2', c=meta_small['U'])
plt.xlabel('Annual Consumption (kWh)')
plt.ylabel('Squared Meters')
plt.title('Annual Consumption vs Squared Meters colored by U_value')
plt.show()

sns.scatterplot(data=meta_small, x='annual_consumption', y='m2', c=meta_small['U'])
plt.xlabel('Annual Consumption (kWh)')
plt.ylabel('Squared Meters')
plt.title('Annual Consumption vs Squared Meters colored by U_value')
plt.show()

# distributions of power consumption when force_off is 0 and when force_off is 1
e_diffs = data['meter_target_df'].groupby(data['force_offs']['force_off_long']).mean()
e_diffs.T.plot.hist(alpha=0.5, bins=200)
plt.xlabel('Average diff in Power Consumption (kWh) between force_off states')
plt.show()

# relative difference in power consumption between force_off states
fig, ax = plt.subplots(1,1 , layout='constrained')
p_rel_change = -e_diffs.diff(axis=0).loc[1.0]/(e_diffs.loc[0.0].values)
p_rel_change.plot.hist(alpha=0.5, bins=200)
plt.xlabel('Relative difference in Power Consumption between force_off states')
ax.spines[['top', 'right']].set_visible(False)
plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# --------------- Try to explain the flexibility with a lightgbm regressor----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
meta_hp = meta_hp.drop(columns=['type', 'profile']).astype(float)
p_rel_change_hp = p_rel_change[meta_hp.index]

# split train test
tr_ratio = 0.8
n_samples = len(p_rel_change_hp)
n_train = int(tr_ratio*n_samples)

meta_hp_tr, meta_hp_te = meta_hp.iloc[:n_train], meta_hp.iloc[n_train:]
p_rel_change_hp_tr, p_rel_change_hp_te = p_rel_change_hp.iloc[:n_train], p_rel_change_hp.iloc[n_train:]


m = LGBMRegressor(n_estimators=200, learning_rate=0.01).fit(meta_hp_tr, p_rel_change_hp_tr)

fig, ax = plt.subplots(1,1 , layout='constrained')
plt.scatter(p_rel_change_hp_te.values, m.predict(meta_hp_te))
plt.plot([0, 1], [0, 1], c='k', ls='--')
plt.xlabel('True relative flexibility')
plt.ylabel('Predicted relative flexibility')
plt.show()

# analyze feature importance of the model using shap
import shap
explainer = shap.Explainer(m)
shap_values = explainer(meta_hp_te)
shap.summary_plot(shap_values, meta_hp_te)


# ----------------------------------------------------------------------------------------------------------------------
# ------------- Investigate flexibility index conditional to external factor (T, h) ------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
x = data['meteo_data']
y = data['meter_target_df'][meta_hp.index]
force_off = data['force_offs']['force_off_long']

flex_call = force_off == 1
nans_idx = ~(pd.isna(y).sum(axis=1)>0) & ~pd.isna(x).any(axis=1)
x = x[nans_idx]
y = y[nans_idx]
flex_call = flex_call[nans_idx]


# insert q-cuts in relevant variable
temps = pd.qcut(x.iloc[:, 0], np.linspace(0, 1, 6))
flex_bins = pd.concat([y, temps, flex_call], axis=1).groupby(['mean_T', 'force_off_long'], observed=False).mean()
flex_bins_rel_ch = (flex_bins.unstack(level=0).diff() / flex_bins.unstack(level=0).loc[False]).loc[True]


fig, ax = plt.subplots(1, 1, layout='constrained', figsize=(4, 10))
# heat map with aspect ratio to auto
sns.heatmap(flex_bins_rel_ch.unstack().sort_values(flex_bins_rel_ch.unstack().columns[0]), ax=ax)
ax.set_xlabel('Temperature bins (°C)')
plt.show()




def time_temp_plot(y, temp_bins, meter_col):
    # Build a tidy frame with: value, temp bin, force flag, hour
    df_hour = pd.DataFrame({
        "value": y[meter_col].values,
        "mean_T": temps.values,                # from your qcut above
        "force_off_long": flex_call.values
    }, index=y.index)

    # Ensure datetime index and add hour-of-day
    if not isinstance(df_hour.index, pd.DatetimeIndex):
        df_hour.index = pd.to_datetime(df_hour.index)
    df_hour["hour"] = df_hour.index.hour

    # Group by (hour, temp_bin, force_flag) and get mean value for this meter
    g = (
        df_hour
        .groupby(["hour", "mean_T", "force_off_long"], observed=False)["value"]
        .mean()
        .unstack("mean_T")                     # columns = temperature bins
    )

    # Split baseline vs flex, compute relative change: (flex - base) / base
    base = g.xs(False, level="force_off_long")
    flex = g.xs(True,  level="force_off_long")

    rel = (flex - base) / base.replace(0, np.nan)   # avoid div-by-zero
    rel.index.name = "Hour of day"
    rel.columns.name = "Temperature bins (°C)"

    # Plot: x = temp bins, y = hour of day
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), layout="constrained")
    sns.heatmap(rel, ax=ax)
    ax.set_xlabel("Temperature bins (°C)")
    ax.set_ylabel("Hour of day")
    ax.set_title(f"Relative change during flex vs baseline — {meter_col}")
    plt.show()

meter_col = 'meter_1680'
time_temp_plot(y, temps, meter_col)
meter_col = 'meter_2639'
time_temp_plot(y, temps, meter_col)

