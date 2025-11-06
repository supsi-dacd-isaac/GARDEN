import numpy as np
from battery_sizing_cfa.cost_functions.lcoe import lcoe_from_results
from battery_sizing_cfa.cost_functions.peak_shaving import peak_shaving_cost_from_results, day_max_cost_from_results, daily_maxima
from battery_sizing_cfa.cost_functions.utils import block_nes_grad_with_h_parallel, build_block_basis, cvar_from_daily_losses, cvar_weighted_mean, frac_rolling_quantile
import pandas as pd
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
      period_peaks_MW=[np.max(p_grid_sim[i*24:(i+1)*24])/1000.0 for i in range(N//24) if len(p_grid_sim[i*24:(i+1)*24]) > 0],
      **specs
    )

    return lcoe_simulation


def rbc_peak_shaving(sampled_pars, L, price, specs, return_adv=False, h=None):
    #lower_threshold = pd.Series(L).rolling(window=int(sampled_pars['n_hours']), min_periods=1).quantile(sampled_pars['lower_q']).to_numpy()
    #upper_threshold = pd.Series(L).rolling(window=int(sampled_pars['n_hours']), min_periods=1).quantile(sampled_pars['higher_q']).to_numpy()

    lower_threshold = frac_rolling_quantile(L, W_star=sampled_pars['n_hours'], q=sampled_pars['lower_q'])
    upper_threshold = frac_rolling_quantile(L, W_star=sampled_pars['n_hours'], q=sampled_pars['higher_q'])

    price_high = price == np.max(price)
    p_battery = specs.get('c_bat_E_kwh', 1.0) * specs.get('energy_ratio', 1.0)
    soc, p_batt, p_grid = rbc_thresholds(L,  # array [T], + = net consumption, - = net production (surplus)
                                         price_high,  # array [T] of 0/1 (unused here, kept for signature compatibility)
                                         capacity_kwh=specs.get('c_bat_E_kwh', 1.0),  # usable capacity [kWh]
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

    d_losses = daily_maxima(p_grid, h)

    peak_cost = cvar_from_daily_losses(d_losses, specs.get('alpha_cvar', 0.9))



    if specs.get('adversarial_perturbation', False):
        adversarial_budget = specs.get('adversarial_budget', 10)
        L_adv = L.copy()

        """
        # retrieve per-timestep peak shaving cost
        peak_cost_base_t = (L + p_batt)**2

        # repeat with marginal increments of the profile to retrieve d(cost)/d(profile)
        L_plus = L*(1.0 + 1e-2) + 1e-2

        soc_plus, p_batt_plus, p_grid_plus = rbc_thresholds(L_plus,  # array [T], + = net consumption, - = net production (surplus)
                                             price_high,  # array [T] of 0/1 (unused here, kept for signature compatibility)
                                             capacity_kwh=specs.get('c_bat_E_kwh', 1.0),  # usable capacity [kWh]
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

        peak_cost_plus_t = (L_plus + p_batt_plus) ** 2

        grads = (peak_cost_plus_t - peak_cost_base_t) / (L * 1e-2 + 1e-2)
        sorted_indices = np.argsort(-grads)
        for a in range(adversarial_budget):
            idx = sorted_indices[a]
            L_adv[idx] += specs.get('adversarial_magnitude', 1e-1) * np.sign(grads[idx])
        

        U = build_block_basis(len(L), block_len=specs.get('adversarial_block_size', 24), stride=24, normalize=False)
        grads = block_nes_grad_with_h_parallel(
            L=L,
            h=h,
            lower_thr=lower_threshold,
            upper_thr=upper_threshold,
            specs=specs,
            U=U,
            sigma=0.3,
            tau=0.5,  # softmax temperature
            day_len=24  # or leave None and use h wrap detection
        )
        sorted_indices = np.argsort(-grads)
        L_adv[sorted_indices[:adversarial_budget]] += specs.get('adversarial_magnitude', 1e-1) * np.sign(grads[sorted_indices[:adversarial_budget]])

        
        f = lambda x: day_max_cost_from_results((x + rbc_thresholds(x, price_high,
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
            )[1]), h=h)

        grads = nes_grad_orthogonal(f, L_adv, sigma=0.3, m=int(50), seed=0)

        sorted_indices = np.argsort(-grads)
        L_adv[sorted_indices[0]] + specs.get('adversarial_magnitude', 1e-1) * np.sign(grads[sorted_indices[0]])
        


        soc_plus, p_batt, p_grid = rbc_thresholds(L_adv,  # array [T], + = net consumption, - = net production (surplus)
                                             price_high,  # array [T] of 0/1 (unused here, kept for signature compatibility)
                                             capacity_kwh=specs.get('c_bat_E_kwh', 1.0),  # usable capacity [kWh]
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
        """


    if return_adv:
        return peak_cost, L_adv
    else:
        return peak_cost



def optimize_pv_battery_rbc(L, PV_base, price, export_price, specs, lims):

    bounds = [
        (0, 3 * lims['x_pv']),  # x_pv
        (0, 3 * lims["E_bat_kWh"]),  # E_bat_kWh
        (0, 100),  # p_threshold
        (0, 1)  # soc_min_price
    ]

    result = differential_evolution(
        rbc_peak_shaving,
        bounds,
        args=(L, PV_base, price, export_price, specs),
        init='random',
        strategy='best1bin',
        maxiter=100,
        popsize=15,
        tol=0.01,
        polish=False
    )



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