import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from matplotlib.pyplot import legend
from scipy.stats import betanbinom
import matplotlib.pyplot as plt

from battery_sizing_cfa.optimizers.deterministic_sizing import optimize_pv_battery
from scipy.optimize import differential_evolution
from battery_sizing_cfa.cost_functions.utils import build_block_basis, cvar_from_daily_losses, frac_rolling_quantile
from battery_sizing_cfa.optimizers.parametric_rbc import rbc_thresholds
from battery_sizing_cfa.optimizers.rbc_sizing import rbc_peak_shaving
from battery_sizing_cfa.optimizers.peak_shaver import mpc_peak_shaving
from battery_sizing_cfa.cost_functions.peak_shaving import daily_maxima, day_max_cost_from_results

def build_covariates(df, H):
    """Build covariates DataFrame from datetime index."""
    covariates = df.copy()
    covariates['hour_of_day'] = df.index.hour
    covariates['day_of_week'] = df.index.dayofweek
    covariates['month_of_year'] = df.index.month
    covariates['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    # add lags of the target variable (past 24 hours)
    for lag in range(1, H+1):
        covariates[f'lag_{lag}'] = df.iloc[:, 0].shift(lag)

    # add target future values (past 24 hours)
    for lead in range(1, H+1):
        covariates[f'lead_{lead}'] = df.iloc[:, 0].shift(-lead)
    covariates = covariates.bfill().ffill()  # handle NaNs from lags
    x = covariates[[c for c in covariates.columns if 'lead_' not in c]]
    y = covariates[[c for c in covariates.columns if 'lead_' in c]]
    return x, y

def train_test_split(df, train_ratio=0.8):
    """Split DataFrame into training and testing sets."""
    n = len(df)
    n_train = int(n * train_ratio)
    df_train = df.iloc[:n_train]
    df_test = df.iloc[n_train:]
    return df_train, df_test

def get_forecasts(df, H=24, train_ratio=0.2):
    x, y = build_covariates(df, H=H)
    x_train, x_test = train_test_split(x, train_ratio=train_ratio)
    y_train, y_test = train_test_split(y, train_ratio=train_ratio)
    m = LGBMRegressor(n_estimators=100, learning_rate=0.1, force_row_wise=True)
    preds = [y_test.iloc[:, 0].values.ravel()]  # first step is just the true value shifted
    preds_tr = [y_train.iloc[:, 0].values.ravel()]
    for step in range(1, H):
        m.fit(x_train, y_train.values[:, step])
        preds.append(m.predict(x_test))
        preds_tr.append(m.predict(x_train))
    preds = np.stack(preds, axis=1)
    preds_tr = np.stack(preds_tr, axis=1)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), layout='constrained')
    ax[0].plot(y_train.values[:, 0], label='train', alpha=1, linewidth=0.5)
    ax[0].plot(y_test.values[:, 0], label='test', alpha=1, linewidth=0.5)
    ax[0].legend()

    ax[1].plot(np.mean(np.abs(y_train.values - preds_tr), axis=0) / np.mean(np.abs(y_train.values)), label='Train')
    ax[1].plot(np.mean(np.abs(y_test.values - preds), axis=0) / np.mean(np.abs(y_test.values)), label='Test')
    ax[1].legend()

    ax[1].set_xlabel('Forecast Horizon (hours)')
    ax[1].set_ylabel('Normalized MAE [-]')
    ax[1].set_title('Normalized MAE per forecast horizon')


    # despine and show
    [ax[0].spines[side].set_visible(False) for side in ['top', 'right']]
    [ax[1].spines[side].set_visible(False) for side in ['top', 'right']]

    plt.show()
    preds = np.vstack([preds_tr[-1], preds[:-1]])
    perfect_preds = np.vstack([y_train.values[-1], y_test.values[:-1]])
    return preds, perfect_preds, x_train, x_test, y_train, y_test

def rbc_sizing(df, specs):

    pass

def one_shot_sizing(L, PV_base, price, export_price, specs):
    res = optimize_pv_battery(
        L=L, PV_base=PV_base, price=price, export_price=export_price,
        solver_options={"time_limit": 120, "threads": 2}, **specs
    )
    return res["E_bat_kWh"], res["P_bat_max_kW"], res["x_pv"]

def optimize_policy_and_run(df_tr, df_te, y_hat_te, specs, control='rbc'):
    L_tr = df_tr.loc[:, 'p_load'].values
    L_te = df_te.loc[:, 'p_load'].values
    price_tr = np.ones_like(L_tr)
    price_te = np.ones_like(L_te)
    if control == 'rbc':
        hour_index_tr = df_tr.index.hour.values
        rbc_peak_shaving_wrap = lambda x, L_tr, price, specs: rbc_peak_shaving({'lower_q':x[0],
                                                                                'higher_q':x[1],
                                                                                'n_hours':x[2]},
                                                                               L_tr, price,
                                                                               {'alpha_cvar': x[3], **specs},
                                                                               h=hour_index_tr)
        bounds = [
            (0, 0.9),  # lower_q
            (0.1, 1),  # higher_q
            (5, 24*7), # n_hours
            (0.0, 1) # alpha_cvar
        ]
        result = differential_evolution(
            rbc_peak_shaving_wrap,
            bounds=bounds,
            args=(L_tr, price_tr, specs),
            init='random',
            strategy='best1bin',
            maxiter=100,
            popsize=20,
            tol=0.01,
            polish=False,
            integrality=(False, False, False)
        )

        # performance on test set
        best_q_low, best_q_high, best_n, best_alpha_cvar = result.x
        L_all = np.concatenate([L_tr, L_te])
        lt_all = frac_rolling_quantile(L_all, W_star=best_n, q=best_q_low)
        lower_threshold = lt_all[-len(L_te):]
        ut_all = frac_rolling_quantile(L_all, W_star=best_n, q=best_q_high)
        upper_threshold = ut_all[-len(L_te):]

        print("Best q_low = {:0.2f}, q_high = {:0.2f}, n_hours = {:0.2f}, alpha_cvar:{:0.2e}".format(best_q_low, best_q_high, best_n, best_alpha_cvar))

        p_battery = specs.get('c_bat_E_kwh', 1.0) * specs.get('energy_ratio', 1.0)

        soc, p_batt, p_grid = rbc_thresholds(
            L_te,  # net consumption array
            L_te * 0,  # placeholder PV flag (kept for signature)
            capacity_kwh=specs.get('c_bat_E_kwh', 1.0),
            soc_start=specs.get('soc_start', 0.5),
            soc_min=specs.get('soc_min', 0.1),
            soc_max=specs.get('soc_max', 0.99),
            p_charge_max=p_battery,
            p_discharge_max=p_battery,
            eta_ch=specs.get('eta_ch', 0.99),
            eta_dis=specs.get('eta_dis', 0.99),
            dt_hours=1.0,
            noise_level=0,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold
        )

    else:
        p_battery = specs.get('c_bat_E_kwh', 1.0) * specs.get('energy_ratio', 1.0)
        e_init = specs.get('soc_start', 0.5) * specs.get('c_bat_E_kwh', 1.0)
        res_mpc = mpc_peak_shaving(y_hat_te, eta_ch=specs['eta_ch'], eta_dis=specs['eta_dis'],
                                               E_max=specs.get('c_bat_E_kwh', 1.0), E_min=0.0,
                                               P_max=p_battery, E_init=e_init)
        p_grid = res_mpc['p_grid']

    return p_grid


def compare_methods(df, sizing_method='one_shot', specs=None, target_name='p_load', H=24, train_ratio=0.8, series=0):
    print('training forecaster...')
    y_hat_te, y_perfect_te, x_train, x_test, y_train, y_test = get_forecasts(df, H=H, train_ratio=train_ratio)
    print('deterministic optimal sizing on {} days...'.format(specs['hours_one_shot_sizing']//24))
    # sizing using one of the methods
    if sizing_method == 'rbc_peak_shaving':
        from battery_sizing_cfa.optimizers.rbc_sizing import rbc_peak_shaving
        # implement RBC sizing logic here
        E_bat_kWh, P_bat_max_kW, x_pv = rbc_sizing(x_train, specs)
    else:
        # implement one-shot sizing logic here
        L = x_train.loc[:, target_name].values[:specs['hours_one_shot_sizing']]
        PV_base = np.zeros_like(L)  # Placeholder for PV generation profile
        #price = np.array([40 if (t % 24) in range(18, 24) else 20 for t in range(len(L))])  # $/MWh
        price = np.ones_like(L) * 20  # $/MWh
        export_price = np.ones_like(L) * 10  # $/MWh
        E_bat_kWh, P_bat_max_kW, x_pv = one_shot_sizing(L, PV_base, price, export_price, specs)

    print('sized battery: {:.2f} kWh, {:.2f} kW, pv size: {:.2f} kW'.format(E_bat_kWh, P_bat_max_kW, x_pv))
    specs.update({'c_bat_E_kwh': E_bat_kWh, 'energy_ratio': E_bat_kWh / P_bat_max_kW, 'x_pv': x_pv})


    PV_base_tr = np.zeros_like(x_train.loc[:, target_name].values)  # Placeholder for PV generation profile
    PV_base_te = np.zeros_like(x_test.loc[:, target_name].values)  # Placeholder for PV generation profile

    ### RBC peak shaving on test set
    from battery_sizing_cfa.optimizers.rbc_sizing import rbc_peak_shaving
    # optimize the RBC policy on the training set
    L_tr = x_train.loc[:, target_name].values + x_pv * PV_base_tr
    L_te = x_test.loc[:, target_name].values + x_pv * PV_base_te

    # run optimized RBC on test set
    print('rbc policy tuning on {} days and testing on {} days...'.format(len(L_tr)//24, len(L_te)//24))
    p_grid_rbc = optimize_policy_and_run(x_train, x_test, y_hat_te, specs, control='rbc')

    # run MPC on test set
    print('mpc policy testing...')
    p_grid_mpc = optimize_policy_and_run(x_train, x_test, y_hat_te, specs, control='mpc')


    # run MPC with perfect forecasts on test set
    print('mpc policy testing with perfect forecasts...')
    #noise =  np.random.randn(y_perfect_te.shape[0]+y_perfect_te.shape[1]-1, 1) * 0.01
    #hankel_noise = np.vstack([noise[i:i+H, 0] for i in range(noise.shape[0]-H+1)])
    p_grid_mpc_opt = optimize_policy_and_run(x_train, x_test, y_perfect_te, specs, control='mpc')

    # retrieve cost on the test set for both methods
    cost_rbc = day_max_cost_from_results(p_grid_rbc, h=x_test.index.hour.values)
    cost_mpc = day_max_cost_from_results(p_grid_mpc, h=x_test.index.hour.values)
    cost_mpc_opt = day_max_cost_from_results(p_grid_mpc_opt, h=x_test.index.hour.values)


    fig, ax = plt.subplots(6, 1, figsize=(12, 8), layout='constrained')
    w_len = len(x_test)//6
    # retrieve standard color map
    colors = plt.get_cmap('tab10')
    for i, a in enumerate(ax.ravel()):
        w = np.arange(w_len) + i * w_len
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.plot(x_test.index.values[w], x_test.iloc[w][target_name].values, label='Load', alpha=1, linewidth=0.5, color=colors(0))
        a.plot(x_test.index.values[w], p_grid_rbc[w], label='RBC', alpha=1, linewidth=0.5, color=colors(1))
        a.plot(x_test.index.values[w], p_grid_mpc[w], label='MPC', alpha=1, linewidth=0.5, color=colors(2))
        a.plot(x_test.index.values[w], p_grid_mpc_opt[w], label='MPC prescient', alpha=1, linewidth=0.5, linestyle='--', color=colors(2))
        a.set_ylabel('Power')
        a.set_title('Grid Power Profiles on Test Set')
        a.legend()
        a.set_xlabel('Time')

    plt.savefig("battery_sizing_cfa/figs/rbc_vs_mpc_profiles_{}.pdf".format( series))

    print(f"Test set cost - RBC: {cost_rbc:.2f}, MPC: {cost_mpc:.2f}, MPC prescient: {cost_mpc_opt:.2f}")


# load spanish data
data = pd.read_pickle("battery_sizing_cfa/datasets/portugal/portugal.pk")
series = 8

df = data.iloc[:24*365, [series]]  # select one series
df.rename(columns={series: 'p_load'}, inplace=True)

train_ratio = 0.5
hours_one_shot_sizing = 24*30
replicate_periods = int(np.ceil(24*365/(hours_one_shot_sizing)))

specs = {'eta_ch': 0.95,
         'eta_dis': 0.95,
         'soc_min':0,
         'soc_max':1,
         'soc_start':0.2,
         'peak_tariff_per_MW_period':500,
         'energy_ratio': 1.0,
         'peak_period_steps':24,
         'c_PV_kw':200,
         'c_bat_E_kwh':120,
         'c_bat_P_kw':50,
         'hours_one_shot_sizing': hours_one_shot_sizing,
         'replicate_periods': replicate_periods}

compare_methods(df, sizing_method='one_shot', specs=specs, target_name='p_load', train_ratio=train_ratio, H=24, series=series)
