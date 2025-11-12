import numpy as np
from battery_sizing_cfa.cost_functions.lcoe import lcoe_from_results
from battery_sizing_cfa.cost_functions.peak_shaving import peak_shaving_cost_from_results, day_max_cost_from_results, daily_maxima
from battery_sizing_cfa.cost_functions.utils import block_nes_grad_with_h_parallel, build_block_basis, cvar_from_daily_losses, cvar_weighted_mean, frac_rolling_quantile
import pandas as pd
from copy import copy
from battery_sizing_cfa.optimizers.parametric_rbc import rbc, rbc_thresholds
from scipy.optimize import differential_evolution

from battery_sizing_cfa.utils.fun_utils import rolling_inverted_quantile_np
from numba import njit

def rbc_opt_fun(sampled_pars, L, PV_base, price, export_price, specs, noise_level=0, **kwargs):

    N = len(L)

    # Unpack parameters, if not enough sampled parameters, use defaults
    pv_level = sampled_pars[0]  # PV size as a fraction of base
    e_batt = sampled_pars[1]    # Battery energy capacity in kWh

    p_threshold = sampled_pars[2]     # Load threshold for threshold-based discharge in k
    soc_min_price = sampled_pars[3]  # Minimum state of charge for price-based discharge (0 to 1)

    price_high = price == np.max(price)
    p_battery = e_batt*specs['energy_ratio']
    peak_period_steps = specs.get('peak_period_steps', 24)

    # Calculate LCOE for the simulation results
    soc_sim, p_batt_sim, p_grid_sim = rbc(
    p_tot=L - pv_level * PV_base,  # net load is total consumption
    price_high=price_high,
    capacity_kwh = e_batt, # Use kWh directly
    soc_min_price_fun=np.array([soc_min_price]), # Ensure it's passed as an array if the simulate_battery expects it
    p_charge_max=p_battery,
    p_discharge_max=p_battery,
    p_threshold_fun=np.array([p_threshold]), # Ensure it's passed as an array if the simulate_battery expects it
    eta_ch=specs['eta_ch'], eta_dis=specs['eta_dis'],
    dt_hours=specs['Delta_t'],
    soc_start=specs['soc_start'],
    soc_min=specs['soc_min'],
    soc_max=specs['soc_max'],
    noise_level=noise_level)

    lcoe_simulation = lcoe_from_results(
      L=L,
      PV_base=PV_base,
      price=price,
      export_price=export_price,
      x_pv=pv_level,  # Use the PV size from optimization
      E_bat_kWh=e_batt, # Use battery energy capacity from optimization
      P_bat_max_kW=p_battery, # Use battery power capacity from optimization
      P_net_kW=p_grid_sim, # Use the net load from the simulation
      # For peak costs in simulation, we need to find the peak import in MW for each period
      period_peaks_MW=[np.max(p_grid_sim[i*peak_period_steps:(i+1)*peak_period_steps])/1000.0 for i in range(N//peak_period_steps) if len(p_grid_sim[i*peak_period_steps:(i+1)*peak_period_steps]) > 0],
      **specs
    )

    return lcoe_simulation


def optimize_lcoe_rbc(sampled_pars, L, PV_base, price, export_price, specs, h=None, noise_level=0, **kwargs):
    specs_temp = copy(specs)
    specs_temp['E_bat_kWh'] = sampled_pars['E_bat_kWh']

    N = len(L)
    # Calculate LCOE for the simulation results
    peak_period_steps = specs_temp.get('peak_period_steps', 24)
    L_eff = L - sampled_pars['pv_level'] * PV_base
    soc_sim, p_batt_sim, p_grid_sim = rbc_peak_shaving(sampled_pars, L_eff, price, specs_temp, return_adv=False, h=h, return_ts=True)
    P_bat_max_kW = sampled_pars['E_bat_kWh'] * specs_temp['energy_ratio']
    lcoe_simulation = lcoe_from_results(
      L=L,
      PV_base=PV_base,
      price=price,
      export_price=export_price,
      x_pv=sampled_pars['pv_level'],  # Use the PV size from optimization
      E_bat_kWh= sampled_pars['E_bat_kWh'], # Use battery energy capacity from optimization
      P_bat_max_kW= P_bat_max_kW, # Use battery power capacity from optimization
      P_net_kW=p_grid_sim, # Use the net load from the simulation
      # For peak costs in simulation, we need to find the peak import in MW for each period
      period_peaks_MW=[np.max(p_grid_sim[i*peak_period_steps:(i+1)*peak_period_steps])/1000.0 for i in range(N//peak_period_steps) if len(p_grid_sim[i*peak_period_steps:(i+1)*peak_period_steps]) > 0],
      Delta_t = specs['Delta_t'],
      c_PV_kw = specs['c_PV_kw'],
      c_bat_E_kWh  = specs['c_bat_E_kWh'],
      c_bat_P_kw = specs['c_bat_P_kw'],
      peak_tariff_per_MW_period = specs['peak_tariff_per_MW_period'],
      peak_period_steps = specs['peak_period_steps'],
      discount_rate =specs['discount_rate'],
      lifetime_years = specs['lifetime_years'],
      replicate_periods = specs['replicate_periods'],
      installation_fixed_costs = specs.get('installation_fixed_costs', 200))

    return lcoe_simulation


def rbc_peak_shaving(sampled_pars, L, price, specs, return_adv=False, h=None, return_ts=False, stratified_cvar=True):
    lower_threshold = frac_rolling_quantile(L, W_star=sampled_pars['n_hours'], q=sampled_pars['lower_q'])
    upper_threshold = frac_rolling_quantile(L, W_star=sampled_pars['n_hours'], q=sampled_pars['higher_q'])

    price_high = price == np.max(price)
    p_battery = specs.get('E_bat_kWh', 1.0) * specs.get('energy_ratio', 1.0)
    soc, p_batt, p_grid = rbc_thresholds(L,  # array [T], + = net consumption, - = net production (surplus)
                                         price_high,  # array [T] of 0/1 (unused here, kept for signature compatibility)
                                         capacity_kwh=specs.get('E_bat_kWh', 1.0),  # usable capacity [kWh]
                                         soc_start=specs.get('soc_start', 0.5),  # initial SOC in [0,1]
                                         soc_min=specs.get('soc_min', 0.1),  # hard minimum SOC (never go below)
                                         soc_max=specs.get('soc_max', 0.99),  # hard maximum SOC (never exceed)
                                         p_charge_max=p_battery,  # max charge power [kW]
                                         p_discharge_max=p_battery,  # max discharge power [kW]
                                         eta_ch=specs.get('eta_ch', 0.99),  # charging efficiency
                                         eta_dis=specs.get('eta_dis', 0.99),  # discharging efficiency
                                         dt_hours=1.0,  # time step [h]
                                         noise_level=0,
                                         lower_threshold=lower_threshold,
                                         upper_threshold=upper_threshold)

    if specs.get('alpha_cvar', 0.9)>0:

        if stratified_cvar:
            # stratified CVaR: compute CVaR per month, then average
            d_losses = daily_maxima(p_grid, h)
            worst_days = []
            for m in np.arange(0, len(d_losses), 30):
                month_worst_k = np.sort(d_losses[m:m + 30])[
                -np.maximum(int(30 * (1 - specs.get('alpha_cvar', 0.9))), 1):]
                worst_days.append(month_worst_k)
            peak_cost = np.mean(np.concatenate(worst_days))
        else:
            d_losses = daily_maxima(p_grid - pd.Series(p_grid).rolling(24 * 7, min_periods=1).mean().values, h)
            peak_cost = np.mean(
                np.sort(d_losses)[-np.maximum(int(len(d_losses) * (1 - specs.get('alpha_cvar', 0.9))), 1):])

        if False:
            import matplotlib.pyplot as plt
            d_maxima = pd.Series(p_grid).groupby(np.arange(len(p_grid)) // 24).max()
            max_locations = d_maxima.index * 24 + d_maxima.index.map(
                lambda x: pd.Series(p_grid)[x * 24:(x + 1) * 24].idxmax() % 24)
            # plot the 10% worst peaks
            worse_indexes = np.argsort(d_maxima)[-int(0.1 * len(d_maxima)):]


            p_grid_norm = pd.Series(p_grid_norm)
            d_maxima = p_grid_norm.groupby(np.arange(len(p_grid_norm))//24).max()
            max_locations_norm = d_maxima.index * 24 + d_maxima.index.map(lambda x: p_grid_norm[x*24:(x+1)*24].idxmax()%24)
            # plot the 10% worst peaks
            worse_indexes_norm = np.argsort(d_maxima)[-int(0.1*len(d_maxima)):]


            fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=True)
            ax.plot(pd.Series(p_grid))
            ax.scatter(max_locations_norm[worse_indexes_norm], pd.Series(p_grid)[max_locations].values[worse_indexes_norm], color='red',
                          label='Daily Maxima', s=20, marker='*')
            ax.scatter(max_locations[worse_indexes], pd.Series(p_grid)[max_locations].values[worse_indexes], color='green',
                          label='Daily Maxima', s=20, marker='o', alpha=0.3)
            plt.show()

    else:
        peak_cost = np.mean(daily_maxima(p_grid,h))

    if specs.get('adversarial_perturbation', False):
        adversarial_budget = specs.get('adversarial_budget', 10)
        L_adv = L.copy()

    if return_ts:
        return soc, p_batt, p_grid
    if return_adv:
        return peak_cost, L_adv
    else:
        return peak_cost




def nes_grad_orthogonal(f, x, sigma=0.3, m=128, seed=0):
    """
    f: callable(x)-> scalar loss
    x: (T,) vector
    sigma: perturbation scale (units of x)
    m: number of directions
    """
    T = x.size
    rng = np.random.default_rng(seed)
    # Build T x m matrix with orthonormal columns via QR
    A = rng.standard_normal((T, m))
    Q, _ = np.linalg.qr(A)                # Q: T x m with orthonormal columns
    U = Q                                 # directions u_i are columns of U

    # Antithetic sampling
    f_plus = np.empty(m, dtype=float)
    f_minus = np.empty(m, dtype=float)
    for i in range(m):
        u = U[:, i]
        f_plus[i]  = f(x + sigma * u)
        f_minus[i] = f(x - sigma * u)

    # (1/(2σm)) Σ (f+ - f-) u
    g = (U @ (f_plus - f_minus)) / (2.0 * sigma * m)
    return g