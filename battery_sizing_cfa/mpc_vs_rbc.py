import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from battery_sizing_cfa.cost_functions.lcoe import lcoe_from_results
import matplotlib.pyplot as plt
from copy import copy
import pickle
from time import time
from battery_sizing_cfa.optimizers.deterministic_sizing import optimize_lcoe_prescient
from scipy.optimize import differential_evolution
from battery_sizing_cfa.cost_functions.utils import build_block_basis, cvar_from_daily_losses, frac_rolling_quantile
from battery_sizing_cfa.optimizers.parametric_rbc import rbc_thresholds
from battery_sizing_cfa.optimizers.rbc_sizing import rbc_peak_shaving, optimize_lcoe_rbc
from battery_sizing_cfa.optimizers.peak_shaver import mpc_peak_shaving
from battery_sizing_cfa.cost_functions.peak_shaving import daily_maxima, day_max_cost_from_results
from battery_sizing_cfa.utils.plot_utils import plot_rbc_vs_mpc_diagnostics, analyze_results_mpc_vs_rbc
import concurrent.futures

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
    m = LGBMRegressor(n_estimators=100, learning_rate=0.1, force_row_wise=True,verbose=-1, n_jobs=1)
    preds = [y_test.iloc[:, 0].values.ravel()]  # first step is just the true value shifted
    preds_tr = [y_train.iloc[:, 0].values.ravel()]
    preds = []  # first step is just the true value shifted
    preds_tr = []
    for step in range(0, H):
        m.fit(x_train, y_train.values[:, step])
        preds.append(m.predict(x_test))
        preds_tr.append(m.predict(x_train))
    preds = np.stack(preds, axis=1)
    preds_tr = np.stack(preds_tr, axis=1)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), layout='constrained')
    ax[0].plot(y_train.values[:, 0], label='train', alpha=1, linewidth=0.5)
    ax[0].plot(y_test.values[:, 0], label='test', alpha=1, linewidth=0.5)
    ax[0].legend()

    norm_mae_tr = np.mean(np.abs(y_train.values - preds_tr), axis=0) / np.mean(np.abs(y_train.values))
    norm_mae_te = np.mean(np.abs(y_test.values - preds), axis=0) / np.mean(np.abs(y_test.values))
    ax[1].plot(norm_mae_tr, label='Train')
    ax[1].plot(norm_mae_te, label='Test')
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
    return preds, perfect_preds, x_train, x_test, y_train, y_test, norm_mae_tr.mean(), norm_mae_te.mean()

def rbc_sizing(L, PV_base, price, export_price, specs, h):
    def rbc_peak_shaving_wrap(x, L, PV_base, price, export_price, specs, h):
        sampled_pars = {'lower_q':x[0],
                        'higher_q':x[1],
                        'n_hours':x[2],
                        'E_bat_kWh':x[3],
                        'pv_level':0}
        specs_temp = copy(specs)
        return optimize_lcoe_rbc(sampled_pars, L, PV_base, price, export_price, specs_temp, h)
    bounds = [
        (0, 0.9),  # lower_q
        (0.1, 1),  # higher_q
        (5, 24 * 7),  # n_hours
        (0, 500.0)  # E_bat_kWh
    ]
    result = differential_evolution(
        rbc_peak_shaving_wrap,
        bounds=bounds,
        args=(L, PV_base, price, export_price, specs, h),
        init='random',
        strategy='best1bin',
        maxiter=300,
        popsize=20,
        tol=0.01,
        polish=False,
        integrality=(False, False, False)
    )
    # cap results from the optimization: if battery size is small put it to 0
    if result.x[3]<1.0:
        result.x[3]=0.0

    return  result.x[3], result.x[3]*specs.get('energy_ratio', 1.0), 0, result.fun, result

def prescient_sizing(L, PV_base, price, export_price, specs):
    res = optimize_lcoe_prescient(
        L=L, PV_base=PV_base, price=price, export_price=export_price,
        solver_options={"time_limit": 120, "threads": 2}, **specs
    )

    return res["E_bat_kWh"], res["P_bat_max_kW"], res["x_pv"], res["LCOE_$/MWh"]

def optimize_policy_and_run(df_tr, df_te, y_hat_te, specs, control='rbc'):
    L_tr = df_tr.loc[:, 'p_load'].values
    L_te = df_te.loc[:, 'p_load'].values
    price_tr = np.ones_like(L_tr)
    price_te = np.ones_like(L_te)
    if control in ['rbc', 'rbc_adv']:
        hour_index_tr = df_tr.index.hour.values
        rbc_peak_shaving_wrap = lambda x, L_tr, price, specs: rbc_peak_shaving({'lower_q':x[0],
                                                                                'higher_q':x[1],
                                                                                'n_hours':x[2]},
                                                                               L_tr, price,
                                                                               {'alpha_cvar': x[3], **specs},
                                                                               h=hour_index_tr)
        alpha_cvar_low_limit = 0.95 if control == 'rbc_adv' else 0
        alpha_cvar_up_limit = 1 if control == 'rbc_adv' else 0
        bounds = [
            (0, 0.9),  # lower_q
            (0.1, 1),  # higher_q
            (5, 24*7), # n_hours
            (alpha_cvar_low_limit, alpha_cvar_up_limit) # alpha_cvar
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

        p_battery = specs.get('E_bat_kWh', 1.0) * specs.get('energy_ratio', 1.0)

        soc, p_batt, p_grid = rbc_thresholds(
            L_te,  # net consumption array
            L_te * 0,  # placeholder PV flag (kept for signature)
            capacity_kwh=specs.get('E_bat_kWh', 1.0),
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
        p_battery = specs.get('E_bat_kWh', 1.0) * specs.get('energy_ratio', 1.0)
        e_init = specs.get('soc_start', 0.5) * specs.get('E_bat_kWh', 1.0)
        res_mpc = mpc_peak_shaving(y_hat_te, eta_ch=specs['eta_ch'], eta_dis=specs['eta_dis'],
                                               E_max=specs.get('E_bat_kWh', 1.0), E_min=0.0,
                                               P_max=p_battery, E_init=e_init)
        p_grid = res_mpc['p_grid']

    return p_grid


def compare_methods(df, sizing_method='prescient', specs=None, target_name='p_load', H=24, train_ratio=0.8, series=0, billing_peak_period_str=None):
    print('training forecaster...')
    y_hat_te, y_perfect_te, x_train, x_test, y_train, y_test, norm_mae_tr, norm_mae_te = get_forecasts(df, H=H, train_ratio=train_ratio)

    if sizing_method == 'rbc_peak_shaving':
        from battery_sizing_cfa.optimizers.rbc_sizing import rbc_peak_shaving
        L = x_train.loc[:, target_name].values
        h = x_train.index.hour.values
        PV_base = np.zeros_like(L)  # Placeholder for PV generation profile
        price = np.ones_like(L) * specs['import_price_static']  # CHF/MWh
        export_price = np.ones_like(L) * specs['export_price_static']  # CHF/MWh
        E_bat_kWh, P_bat_max_kW, x_pv, lcoe_sizing, res_opt_sizing = rbc_sizing(L, PV_base, price, export_price, specs, h=h)
    else:
        print('deterministic optimal sizing on {} days...'.format(specs['hours_prescient_sizing'] // 24))
        L = x_train.loc[:, target_name].values[:specs['hours_prescient_sizing']]
        PV_base = np.zeros_like(L)  # Placeholder for PV generation profile
        price = np.ones_like(L) * specs['import_price_static']  # CHF/MWh
        export_price = np.ones_like(L) * specs['export_price_static']  # CHF/MWh
        E_bat_kWh, P_bat_max_kW, x_pv, lcoe_sizing = prescient_sizing(L, PV_base, price, export_price, specs)

    print('sized battery: {:.2f} kWh, {:.2f} kW, pv size: {:.2f} kW'.format(E_bat_kWh, P_bat_max_kW, x_pv))
    if E_bat_kWh <=0:
        print('BATTERY SIZE IS 00000000000000000000000000000000000000000000000000000')
    specs.update({'E_bat_kWh': E_bat_kWh, 'energy_ratio': P_bat_max_kW / E_bat_kWh  if P_bat_max_kW>0 else 0, 'x_pv': x_pv})


    PV_base_tr = np.zeros_like(x_train.loc[:, target_name].values)  # Placeholder for PV generation profile
    PV_base_te = np.zeros_like(x_test.loc[:, target_name].values)  # Placeholder for PV generation profile

    ### RBC peak shaving on test set
    from battery_sizing_cfa.optimizers.rbc_sizing import rbc_peak_shaving
    # optimize the RBC policy on the training set
    L_tr = x_train.loc[:, target_name].values + x_pv * PV_base_tr
    L_te = x_test.loc[:, target_name].values + x_pv * PV_base_te

    # run optimized RBC on test set
    if sizing_method == 'rbc_peak_shaving':
        print('rbc policy using pars from sizing period, testing on {} days...'.format(len(L_te) // 24))
        lower_q, higher_q, n_hours = res_opt_sizing.x[:3]
        L_all = np.concatenate([L_tr, L_te])
        lt_all = frac_rolling_quantile(L_all, W_star=n_hours, q=lower_q)
        lower_threshold = lt_all[-len(L_te):]
        ut_all = frac_rolling_quantile(L_all, W_star=n_hours, q=higher_q)
        upper_threshold = ut_all[-len(L_te):]

        soc, p_batt, p_grid_rbc = rbc_thresholds(
            L_te,  # net consumption array
            L_te * 0,  # placeholder PV flag (kept for signature)
            capacity_kwh=specs.get('E_bat_kWh', 1.0),
            soc_start=specs.get('soc_start', 0.5),
            soc_min=specs.get('soc_min', 0.1),
            soc_max=specs.get('soc_max', 0.99),
            p_charge_max=P_bat_max_kW,
            p_discharge_max=P_bat_max_kW,
            eta_ch=specs.get('eta_ch', 0.99),
            eta_dis=specs.get('eta_dis', 0.99),
            dt_hours=1.0,
            noise_level=0,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold
        )
    else:
        print('rbc policy tuning on {} days and testing on {} days...'.format(len(L_tr)//24, len(L_te)//24))
        p_grid_rbc = optimize_policy_and_run(x_train, x_test, y_hat_te, specs, control='rbc')

    print('rbc policy tuning on {} days and testing on {} days...'.format(len(L_tr)//24, len(L_te)//24))
    p_grid_rbc_adv = optimize_policy_and_run(x_train, x_test, y_hat_te, specs, control='rbc_adv')

    # run MPC on test set
    print('mpc policy testing...')
    p_grid_mpc = optimize_policy_and_run(x_train, x_test, y_hat_te, specs, control='mpc')


    # run MPC with perfect forecasts on test set
    print('mpc policy testing with perfect forecasts...')
    #noise =  np.random.randn(y_perfect_te.shape[0]+y_perfect_te.shape[1]-1, 1) * 0.01
    #hankel_noise = np.vstack([noise[i:i+H, 0] for i in range(noise.shape[0]-H+1)])
    p_grid_mpc_opt = optimize_policy_and_run(x_train, x_test, y_perfect_te, specs, control='mpc')

    results = {
        'lcoe_sizing': lcoe_sizing,
        'target_name': target_name,
        'time_index': x_test.index,
        'mean_abs_val': x_test.loc[:, target_name].abs().mean() + 1e-3,
        'profiles': {
            'rbc': p_grid_rbc,
            'rbc_adv': p_grid_rbc_adv,
            'mpc': p_grid_mpc,
            'mpc_opt': p_grid_mpc_opt,
            'no_battery': x_test.loc[:, target_name].values + x_pv * PV_base_te
        },
        'E_bat_kWh': E_bat_kWh,
        'P_bat_max_kW': P_bat_max_kW,
        'x_pv': x_pv,
        'sizing_method':sizing_method,
        'norm_mae_tr': norm_mae_tr,
        'norm_mae_te': norm_mae_te
    }

    results['costs_daily_max'] = {k: day_max_cost_from_results(v, h=x_test.index.hour) for k, v in results['profiles'].items()}
    results['daily_maxima'] = {k: daily_maxima(v, h=x_test.index.hour) for k, v in results['profiles'].items()}

    replicate_periods = 365*24//(len(x_test))
    pps = specs['peak_period_steps']
    price_te = np.ones_like(L_te) * specs['import_price_static']
    export_price_te = np.ones_like(L_te) * specs['export_price_static']
    results['lcoe'] = {k: lcoe_from_results(
      L=L_te,
      PV_base=PV_base_te,
      price=price_te,
      export_price=export_price_te,
      x_pv=specs['x_pv'] if k is not 'no_battery' else 0,  # Use the PV size from optimization
      E_bat_kWh= specs['E_bat_kWh'] if k is not 'no_battery' else 0, # Use battery energy capacity from optimization
      P_bat_max_kW= P_bat_max_kW if k is not 'no_battery' else 0, # Use battery power capacity from optimization
      P_net_kW=v, # Use the net load from the simulation
      # For peak costs in simulation, we need to find the peak import in MW for each period
      period_peaks_MW=[np.max(v[i*pps:(i+1)*pps])/1000.0 for i in range(len(v)//pps) if len(v[i*pps:(i+1)*pps]) > 0],
      Delta_t = specs['Delta_t'],
      c_PV_kw = specs['c_PV_kw'],
      c_bat_E_kWh  = specs['c_bat_E_kWh'],
      c_bat_P_kw = specs['c_bat_P_kw'],
      peak_tariff_per_MW_period = specs['peak_tariff_per_MW_period'],
      peak_period_steps = pps,
      discount_rate =specs['discount_rate'],
      lifetime_years = specs['lifetime_years'],
      replicate_periods = replicate_periods,
      installation_fixed_costs = specs.get('installation_fixed_costs', 200))
      for k, v in results['profiles'].items()}


    plot_rbc_vs_mpc_diagnostics(x_test, results, series, billing_peak_period_str, sizing_method)


    print('Test set costs')
    [print('{}:{:0.2e}'.format(k, v)) for k, v in results['lcoe'].items()]

    return results

# load spanish data


hours_prescient_sizing = 24*30
train_ratio = 0.5

specs = {'eta_ch': 0.97,
         'eta_dis': 0.97,
         'soc_min':0,
         'soc_max':1,
         'soc_start':0.2,
         'peak_tariff_per_MW_period':5750/30,
         'energy_ratio': 1.0,
         'peak_period_steps':24,
         'c_PV_kw':200,
         'c_bat_E_kWh':120,
         'c_bat_P_kw':50,
         'hours_prescient_sizing': hours_prescient_sizing,
         'replicate_periods': int(np.floor(24*365/(hours_prescient_sizing))),
         'Delta_t':1.0,
         'discount_rate':0.06,
         'lifetime_years':15.0,
         'billing_peak_period_str':'daily',
         'sizing_method':'prescient',
         'import_price_static':200,
         'export_price_static':50,
         'installation_fixed_costs':1000
         }


def run(data, n_profiles, sizing_method, specs, train_ratio):
    results = {}
    # Parallelize per-series processing using multiple processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future_map = {}
        for series in range(n_profiles):
            print('Processing series {}'.format(series))
            t0 = time()
            df = data.iloc[:24*365, [series]].copy()  # select one series
            df.rename(columns={series: 'p_load'}, inplace=True)
            # submit job
            fut = executor.submit(
                compare_methods,
                df,
                sizing_method,
                copy(specs),
                'p_load',
                24,
                train_ratio,
                series,
                specs['billing_peak_period_str']
            )
            future_map[fut] = (series, t0)
        for fut in concurrent.futures.as_completed(future_map):
            series, t0 = future_map[fut]
            try:
                results[series] = fut.result()
                print('Series {} processed in {:.2f} seconds'.format(series, time()-t0))
            except Exception as e:
                print(f'Series {series} failed: {e}')
    # save results to file

    res_path = "battery_sizing_cfa/results/rbc_vs_mpc_results_{}_peaks_billed_{}.pk".format(sizing_method, specs['billing_peak_period_str'])
    with open(res_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    # post-analysis and plotting
    analyze_results_mpc_vs_rbc(res_path, specs)


# First, run with daily peak periods, use one-shot sizing
data = pd.read_pickle("battery_sizing_cfa/datasets/portugal/portugal.pk")

# take into account the data was wrongly scaled before
data = data / 4.0 # BEWARE! THIS IS SUPER SPECIFIC TO THE PORTUGAL DATASET!!!

#specs.update({'peak_period_steps':24})
#run(data, n_profiles=100, sizing_method='prescient', specs=specs, train_ratio=train_ratio)
#plt.show()

# Then, run with monthly peak periods, use RBC sizing


specs.update({'peak_period_steps':24*30,
              'peak_tariff_per_MW_period':5750,
              'replicate_periods': int(np.floor(24*365/(hours_prescient_sizing))),
              #'replicate_periods': int(np.ceil(24*365/(24*365 * train_ratio))),
              'billing_peak_period_str':'monthly'})
run(data, n_profiles=100, sizing_method='prescient', specs=specs, train_ratio=train_ratio)
plt.show()

# Then, run with monthly peak periods, use RBC sizing
specs['sizing_method'] = 'rbc_peak_shaving'
specs.update({'peak_period_steps':24*30,
              'peak_tariff_per_MW_period':5750,
              #'replicate_periods': int(np.ceil(24*365/(hours_prescient_sizing)))})
              'replicate_periods': int(np.floor(24*365/(24*365 * train_ratio))),
              'billing_peak_period_str':'monthly'})

run(data, n_profiles=100, sizing_method='rbc_peak_shaving', specs=specs, train_ratio=train_ratio)
plt.show()