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
from battery_sizing_cfa.optimizers.parametric_rbc import rbc_thresholds, rbc_thresholds_dyn_price
from battery_sizing_cfa.optimizers.rbc_sizing import rbc_peak_shaving, optimize_lcoe_rbc, rbc_dyn_price
from battery_sizing_cfa.optimizers.peak_shaver import mpc_peak_shaving, mpc_dyn_tariff
from battery_sizing_cfa.cost_functions.peak_shaving import daily_maxima, day_max_cost_from_results
from battery_sizing_cfa.utils.plot_utils import plot_rbc_vs_mpc_diagnostics, analyze_results_mpc_vs_rbc
import concurrent.futures
from tqdm import tqdm

def build_covariates(df, H, x_pv=0.0):
    """Build covariates DataFrame from datetime index."""
    covariates = df.copy()
    covariates['hour_of_day'] = df.index.hour
    covariates['day_of_week'] = df.index.dayofweek
    covariates['month_of_year'] = df.index.month
    covariates['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    # add lags of the target variable (past 24 hours)
    new_covs = {}
    for lag in range(1, H+1):
        new_covs[f'lag_{lag}'] = df.loc[:, 'p_load'].shift(lag) - x_pv * df.loc[:, 'pv_base'].shift(lag)

    # add target future values (past 24 hours)
    for lead in range(1, H+1):
        new_covs[f'lead_{lead}'] = df.loc[:, 'p_load'].shift(-lead) - x_pv * df.loc[:, 'pv_base'].shift(-lead)
    covariates = pd.concat([covariates, pd.concat(new_covs, axis=1)], axis=1)
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

def get_forecasts(df, H=96, train_ratio=0.2, preformat=False, lightgbm=True, x_pv=0):
    x, y = build_covariates(df, H=H, x_pv=x_pv)
    x_train, x_test = train_test_split(x, train_ratio=train_ratio)
    y_train, y_test = train_test_split(y, train_ratio=train_ratio)
    if preformat:
        return x_train, x_test, y_train, y_test
    preds = [y_test.iloc[:, 0].values.ravel()]  # first step is just the true value shifted
    preds_tr = [y_train.iloc[:, 0].values.ravel()]
    if lightgbm:
        m = LGBMRegressor(n_estimators=100, learning_rate=0.1, min_child_samples=50, force_row_wise=True, verbose=-1, n_jobs=1)
        for step in tqdm(range(1, H)):
            m.fit(x_train, y_train.values[:, step])
            preds.append(m.predict(x_test))
            preds_tr.append(m.predict(x_train))
        preds = np.stack(preds, axis=1)
        preds_tr = np.stack(preds_tr, axis=1)

    else:
        from sklearn.linear_model import LinearRegression
        m = LinearRegression(fit_intercept=True).fit(x_train, y_train.values)
        preds = m.predict(x_test)
        preds_tr = m.predict(x_train)
        preds[:, 0] = y_test.values[:, 0]
        preds_tr[:, 0] = y_train.values[:, 0]


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
                        'pv_level':x[4] if 'size_pv' in specs and specs['size_pv'] is True else 0.0}
        specs_temp = copy(specs)
        return optimize_lcoe_rbc(sampled_pars, L, PV_base, price, export_price, specs_temp, h)
    bounds = [
        (0, 0.9),  # lower_q
        (0.1, 1),  # higher_q
        (5, STEPS_PER_DAY * 7),  # n_hours
        (0, 500.0)  # E_bat_kWh
    ]
    if 'size_pv' in specs and specs['size_pv'] is True:
        bounds.append((0.0, 500.0))  # pv_level

    result = differential_evolution(
        rbc_peak_shaving_wrap,
        bounds=bounds,
        args=(L, PV_base, price, export_price, specs, h),
        init='random',
        strategy='best1bin',
        maxiter=300,
        popsize=20,
        tol=0.01,
        polish=False
    )
    # cap results from the optimization: if battery size is small put it to 0
    if result.x[3]<1.0:
        result.x[3]=0.0
    pv_size_opt = 0.0 if ('size_pv' not in specs or specs['size_pv'] is not True) else result.x[4]
    return  result.x[3], result.x[3]*specs.get('energy_ratio', 1.0), 0, result.fun, result, pv_size_opt

def prescient_sizing(L, PV_base, price, export_price, specs):
    res = optimize_lcoe_prescient(
        L=L, PV_base=PV_base, price=price, export_price=export_price,
        solver_options={"time_limit": 120, "threads": 2}, **specs
    )

    return res["E_bat_kWh"], res["P_bat_max_kW"], res["x_pv"], res["LCOE_$/MWh"]

def optimize_policy_and_run(x_tr, x_te, y_hat_te, specs, control='rbc'):

    price_buy_tr = x_tr['import_price_profile'].values  # CHF/MWh
    price_buy_te = x_te['import_price_profile'].values  # CHF/MWh
    price_sell_tr = x_tr['export_price_profile'].values  # CHF/MWh
    price_sell_te = x_te['export_price_profile'].values  # CHF/MWh

    p_tr = x_tr.loc[:, 'p_load'].values - x_tr.loc[:, 'pv'].values
    p_te = x_te.loc[:, 'p_load'].values - x_te.loc[:, 'pv'].values
    if control in ['rbc', 'rbc_adv']:
        hour_index_tr = x_tr.index.hour.values
        if specs['tariff_scheme'] == 'peak_shaving':
            rbc_wrap = lambda x, L, price, specs: rbc_peak_shaving({'lower_q':x[0],
                                                                                'higher_q':x[1],
                                                                                'n_hours':x[2]},
                                                                                 L, price,
                                                                                {'alpha_cvar': x[3], **specs},
                                                                                 h=hour_index_tr)
        else:
            rbc_wrap = lambda x, L, price, specs: rbc_dyn_price({'lower_q':x[0],
                                                                            'higher_q':x[1],
                                                                            'n_hours':x[2]},
                                                                            L, price,
                                                                            {'alpha_cvar': x[3], **specs},
                                                                            h=hour_index_tr)
        alpha_cvar_low_limit = 0.5 if control == 'rbc_adv' else 0
        alpha_cvar_up_limit = 1 if control == 'rbc_adv' else 0
        bounds = [
            (0, 0.9),  # lower_q
            (0.1, 1),  # higher_q
            (5, STEPS_PER_DAY*7), # n_hours
            (alpha_cvar_low_limit, alpha_cvar_up_limit) # alpha_cvar
        ]
        result = differential_evolution(
            rbc_wrap,
            bounds=bounds,
            args=(p_tr, price_buy_tr, specs),
            init='random',
            strategy='best1bin',
            maxiter=100,
            popsize=20,
            tol=0.01,
            polish=False
        )

        # performance on test set
        best_q_low, best_q_high, best_n, best_alpha_cvar = result.x
        L_all = np.concatenate([p_tr, p_te])
        price_all = np.concatenate([price_buy_tr, price_buy_te])
        if specs['tariff_scheme'] == 'peak_shaving':
            v = L_all
        else:
            v = L_all * price_all
        # lower threshold is computed on power, uppper could be power or power*price
        lt_all = frac_rolling_quantile(L_all, W_star=best_n, q=best_q_low)
        lower_threshold = lt_all[-len(p_te):]
        ut_all = frac_rolling_quantile(v, W_star=best_n, q=best_q_high)
        upper_threshold = ut_all[-len(p_te):]

        print("Best q_low = {:0.2f}, q_high = {:0.2f}, n_hours = {:0.2f}, alpha_cvar:{:0.2e}".format(best_q_low, best_q_high, best_n, best_alpha_cvar))

        p_battery = specs.get('E_bat_kWh', 1.0) * specs.get('energy_ratio', 1.0)

        if specs['tariff_scheme'] == 'peak_shaving':

            soc, p_batt, p_grid = rbc_thresholds(
                p_te,  # net consumption array
                p_te * 0,  # placeholder PV flag (kept for signature)
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
            soc, p_batt, p_grid = rbc_thresholds_dyn_price(
                p_te,  # net consumption array
                price_buy_te,  # placeholder PV flag (kept for signature)
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
        if specs['tariff_scheme'] == 'peak_shaving':
            res_mpc = mpc_peak_shaving(y_hat_te, eta_ch=specs['eta_ch'], eta_dis=specs['eta_dis'],
                                                   E_max=specs.get('E_bat_kWh', 1.0), E_min=0.0,
                                                   P_max=p_battery, E_init=e_init)
        else:
            res_mpc = mpc_dyn_tariff(y_hat_te, p_sell=price_sell_te, p_buy=price_buy_te,
                                     eta_ch=specs['eta_ch'], eta_dis=specs['eta_dis'],
                                     E_max=specs.get('E_bat_kWh', 1.0), E_min=0.0,
                                     P_max=p_battery, E_init=e_init)

        p_grid = res_mpc['p_grid']

    return p_grid


def compare_methods(df, sizing_method='prescient', specs=None, target_name='p_load', H=96, train_ratio=0.8, series=0, billing_peak_period_str=None):
    print('training forecaster...')
    x_train, x_test, y_train, y_test = get_forecasts(df, H=H, train_ratio=train_ratio, preformat=True)

    PV_base_tr = df.loc[x_train.index, 'pv_base'].values
    if sizing_method == 'rbc_peak_shaving':
        from battery_sizing_cfa.optimizers.rbc_sizing import rbc_peak_shaving
        L = x_train.loc[:, target_name].values
        h = x_train.index.hour.values
        price_tr = x_train['import_price_profile'].values  # CHF/MWh
        export_price_tr = x_train['export_price_profile'].values  # CHF/MWh
        E_bat_kWh, P_bat_max_kW, x_pv, lcoe_sizing, res_opt_sizing, pv_size_opt = rbc_sizing(L, PV_base_tr, price_tr, export_price_tr, specs, h=h)
    else:
        print('deterministic optimal sizing on {} days...'.format(specs['steps_prescient_sizing'] // STEPS_PER_DAY))
        L = x_train.loc[:, target_name].values[:specs['steps_prescient_sizing']]
        price_tr = x_train['import_price_profile'].values[:specs['steps_prescient_sizing']]  # CHF/MWh
        export_price_tr = x_train['export_price_profile'].values[:specs['steps_prescient_sizing']]  # CHF/MWh
        PV_base_tr = PV_base_tr[:specs['steps_prescient_sizing']]
        E_bat_kWh, P_bat_max_kW, x_pv, lcoe_sizing = prescient_sizing(L, PV_base_tr, price_tr, export_price_tr, specs)

    print('sized battery: {:.2f} kWh, {:.2f} kW, pv size: {:.2f} kW'.format(E_bat_kWh, P_bat_max_kW, x_pv))

    specs.update({'E_bat_kWh': E_bat_kWh, 'energy_ratio': P_bat_max_kW / E_bat_kWh  if P_bat_max_kW>0 else 0, 'x_pv': x_pv})

    ### RBC peak shaving on test set
    print('training forecaster for control policies...')
    df_aug = df.copy()
    df_aug['pv'] = df['pv_base'] * x_pv
    #df_aug['p_load'] = df[target_name] - df_aug['pv']
    y_hat_te, y_perfect_te, x_train, x_test, y_train, y_test, norm_mae_tr, norm_mae_te = get_forecasts(df_aug, H=H, train_ratio=train_ratio, preformat=False, x_pv=x_pv)
    # extract relevant arrays
    L_tr, L_te = x_train.loc[:, target_name].values, x_test.loc[:, target_name].values
    price_buy_tr, price_buy_te = x_train['import_price_profile'].values, x_test['import_price_profile'].values
    price_sell_tr, price_sell_te = x_train['export_price_profile'].values, x_test['export_price_profile'].values
    # run optimized RBC on test set
    if E_bat_kWh <=0:
        print('BATTERY SIZE IS 00000000000000000000000000000000000000000000000000000')
        p_grid_rbc = x_test.loc[:, target_name].values
        p_grid_rbc_adv = x_test.loc[:, target_name].values
        p_grid_mpc = x_test.loc[:, target_name].values
        p_grid_mpc_opt = x_test.loc[:, target_name].values
    else:
        if sizing_method == 'rbc_peak_shaving':
            print('rbc policy using pars from sizing period, testing on {} days...'.format(len(L_te) // STEPS_PER_DAY))
            lower_q, higher_q, n_hours = res_opt_sizing.x[:3]
            pv_all = np.concatenate([x_train.loc[:, 'pv'].values, x_test.loc[:, 'pv'].values])
            p_all = np.concatenate([L_tr, L_te]) - pv_all
            price_all = np.concatenate([price_buy_tr, price_buy_te])
            if specs['tariff_scheme'] == 'peak_shaving':
                v = p_all
            else:
                v = p_all * price_all
            lt_all = frac_rolling_quantile(p_all, W_star=n_hours, q=lower_q)
            lower_threshold = lt_all[-len(L_te):]
            ut_all = frac_rolling_quantile(v, W_star=n_hours, q=higher_q)
            upper_threshold = ut_all[-len(L_te):]
            p_all_te = L_te - x_test.loc[:, 'pv'].values
            if specs['tariff_scheme'] == 'peak_shaving':
                soc, p_batt, p_grid_rbc = rbc_thresholds(
                    p_all_te,  # net consumption array
                    p_all_te*0,  # placeholder PV flag (kept for signature)
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
                soc, p_batt, p_grid = rbc_thresholds_dyn_price(
                    p_all_te,  # net consumption array
                    price_buy_te,  # placeholder PV flag (kept for signature)
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
            print('rbc policy tuning on {} days and testing on {} days...'.format(len(L_tr)//STEPS_PER_DAY, len(L_te)//STEPS_PER_DAY))
            p_grid_rbc = optimize_policy_and_run(x_train, x_test, y_hat_te, specs, control='rbc')

        print('rbc policy tuning on {} days and testing on {} days...'.format(len(L_tr)//STEPS_PER_DAY, len(L_te)//STEPS_PER_DAY))
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
            'baseline': x_test.loc[:, target_name].values
        },
        'buying_prices': x_test['import_price_profile'].values,
        'selling_prices': x_test['export_price_profile'].values,
        'E_bat_kWh': E_bat_kWh,
        'P_bat_max_kW': P_bat_max_kW,
        'x_pv': x_pv,
        'sizing_method':sizing_method,
        'norm_mae_tr': norm_mae_tr,
        'norm_mae_te': norm_mae_te
    }

    results['costs_daily_max'] = {k: day_max_cost_from_results(v, h=x_test.index.hour) for k, v in results['profiles'].items()}
    results['daily_maxima'] = {k: daily_maxima(v, h=x_test.index.hour) for k, v in results['profiles'].items()}

    replicate_periods = 365*STEPS_PER_DAY//(len(x_test))
    pps = specs['peak_period_steps']
    price_te = x_test['import_price_profile'].values
    export_price_te = x_test['export_price_profile'].values
    results['lcoe'] = {k: lcoe_from_results(
      L=L_te,
      PV_base=df.loc[x_test.index, 'pv_base'].values,
      price=price_te,
      export_price=export_price_te,
      x_pv=specs['x_pv'] if k != 'baseline' else 0,  # Use the PV size from optimization
      E_bat_kWh= specs['E_bat_kWh'] if k != 'baseline' else 0, # Use battery energy capacity from optimization
      P_bat_max_kW= P_bat_max_kW if k != 'baseline' else 0, # Use battery power capacity from optimization
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

STEPS_PER_DAY = 24
Delta_t = 24/STEPS_PER_DAY
steps_prescient_sizing = STEPS_PER_DAY*30
train_ratio = 0.5

specs = {'eta_ch': 0.97,
         'eta_dis': 0.97,
         'soc_min':0,
         'soc_max':1,
         'soc_start':0.2,
         'peak_tariff_per_MW_period':5750/30,
         'energy_ratio': 1.0,
         'peak_period_steps':STEPS_PER_DAY,
         'c_PV_kw':1500,
         'c_bat_E_kWh':120,
         'c_bat_P_kw':50,
         'steps_prescient_sizing': steps_prescient_sizing,
         'replicate_periods': int(np.floor(STEPS_PER_DAY*365/(steps_prescient_sizing))),
         'Delta_t':Delta_t,
         'discount_rate':0.06,
         'lifetime_years':15.0,
         'billing_peak_period_str':'daily',
         'sizing_method':'prescient',
         'installation_fixed_costs':1000,
         'size_pv':True,
         'tariff_scheme':'dynamic_prices' # 'dynamic_prices' or 'peak_shaving'
         }


def run(powers, pv_bases, import_price, export_price, n_profiles, sizing_method, specs, train_ratio):
    results = {}
    # Parallelize per-series processing using multiple processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future_map = {}
        for series in range(n_profiles):
            print('Processing series {}'.format(series))
            t0 = time()
            s = powers.iloc[:, [series]].copy()  # select one series
            s.rename(columns={series: 'p_load'}, inplace=True)
            pv = pv_bases.iloc[:, [series]].rename(columns={series: 'pv_base'})
            df = pd.concat([s, pv, import_price, export_price], axis=1)

            # submit job
            fut = executor.submit(
                compare_methods,
                df,
                sizing_method,
                copy(specs),
                'p_load',
                STEPS_PER_DAY,
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


# Load SWISS DSO dataset

powers = pd.read_pickle("battery_sizing_cfa/datasets/swiss_dso/power.pk")
pv_bases = pd.read_pickle("battery_sizing_cfa/datasets/swiss_dso/solar.pk")
stats = pd.read_pickle("battery_sizing_cfa/datasets/swiss_dso/stats.pk")
tariff = pd.read_csv('battery_sizing_cfa/datasets/swiss_dso/tariffa_dinamica.csv')


# Parse timestamps
tariff["ts"] = pd.to_datetime(tariff["Unnamed: 0"])
tariff["ts"] = pd.to_datetime(tariff["ts"])

# Make tz-aware (assume UTC)
tariff["ts"] = tariff["ts"].dt.tz_localize("UTC")
delta = powers.index[0] - tariff["ts"].min()
tariff["ts_aligned"] = tariff["ts"] + delta
tariff = tariff.set_index("ts_aligned").sort_index()
tariff_trimmed = tariff.loc[powers.index.min():powers.index.max()]
tariff_aligned = tariff_trimmed.reindex(powers.index)
tariff_aligned.drop(['ts', 'Unnamed: 0'], axis=1, inplace=True)


import_price = (31.17- 10.83 + tariff_aligned) * 1.081 # cents/kWh
import_price *= 10 # CHF/MWh
export_price = import_price*0 + 50.0  # 5 CHF/MWh export price
import_price.rename(columns={'tariffa_dinamica': 'import_price_profile'}, inplace=True)
export_price.rename(columns={'tariffa_dinamica': 'export_price_profile'}, inplace=True)

pv_bases = pv_bases / stats['installed_power'].values[:pv_bases.shape[1]]

# downsample everything to hourly
powers = powers.resample('1h').mean()
pv_bases = pv_bases.resample('1h').mean()
import_price = import_price.resample('1h').mean()
export_price = export_price.resample('1h').mean()

specs.update({'peak_period_steps':STEPS_PER_DAY})
run(powers, pv_bases, import_price, export_price, n_profiles=100, sizing_method='prescient', specs=specs, train_ratio=train_ratio)
#plt.show()

# Then, run with monthly peak periods, use RBC sizing

specs.update({'peak_period_steps':STEPS_PER_DAY*30,
              'peak_tariff_per_MW_period':5750,
              'replicate_periods': int(np.floor(STEPS_PER_DAY*365/(steps_prescient_sizing))),
              #'replicate_periods': int(np.ceil(STEPS_PER_DAY*365/(STEPS_PER_DAY*365 * train_ratio))),
              'billing_peak_period_str':'monthly'})
run(powers, pv_bases, import_price, export_price, n_profiles=100, sizing_method='prescient', specs=specs, train_ratio=train_ratio)
plt.show()

# Then, run with monthly peak periods, use RBC sizing
specs['sizing_method'] = 'rbc_peak_shaving'
specs.update({'peak_period_steps':STEPS_PER_DAY*30,
              'peak_tariff_per_MW_period':5750,
              #'replicate_periods': int(np.ceil(STEPS_PER_DAY*365/(steps_prescient_sizing)))})
              'replicate_periods': int(np.floor(STEPS_PER_DAY*365/(STEPS_PER_DAY*365 * train_ratio))),
              'billing_peak_period_str':'monthly'})

run(powers, pv_bases, import_price, export_price, n_profiles=100, sizing_method='rbc_peak_shaving', specs=specs, train_ratio=train_ratio)
plt.show()