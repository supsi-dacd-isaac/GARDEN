import numpy as np
from battery_sizing_cfa.cost_functions.lcoe import lcoe_from_results
from battery_sizing_cfa.cost_functions.peak_shaving import peak_shaving_cost_from_results
import pandas as pd
from battery_sizing_cfa.optimizers.parametric_rbc import rbc, rbc_thresholds
from battery_sizing_cfa.utils.fun_utils import rolling_inverted_quantile_np

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


def rbc_peak_shaving(sampled_pars, L, price, specs):
    lower_threshold = pd.Series(L).rolling(window=sampled_pars['n_hours'], min_periods=1).quantile(sampled_pars['lower_q']).to_numpy()
    upper_threshold = pd.Series(L).rolling(window=sampled_pars['n_hours'], min_periods=1).quantile(sampled_pars['higher_q']).to_numpy()


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

    if specs.get('adversarial_perturbation', False):
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

        peak_cost_plus_t = (L_plus + p_batt_plus)**2
        peak_cost_derivative = (peak_cost_plus_t - peak_cost_base_t) / (L * 1e-2 + 1e-2)
        # sort by magnitude of increment
        sorted_indices = np.argsort(-np.abs(peak_cost_derivative))
        # add perturbations in order of impact until adversarial budget is exhausted
        adversarial_budget = specs.get('adversarial_budget', 10)
        """
        m=50
        sigma=0.1
        vals = []
        xis = []
        for _ in range(m):
            xi = np.random.randn(*L.shape)
            Lp = L + sigma * xi
            soc_p, p_batt_p, _ = rbc_thresholds(
                Lp, price_high,
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
                upper_threshold=upper_threshold,
            )
            # pick your scalar objective; here: sum of squared grid power proxy
            val = ((Lp + p_batt_p) ** 2).mean()
            vals.append(val);
            xis.append(xi)
        vals = np.array(vals);
        xis = np.stack(xis, axis=0)
        vals = (vals - vals.mean()) / (vals.std() + 1e-8)  # variance reduction
        g_hat = (vals[:, None] * xis).mean(axis=0) / sigma
        """
        # 10% increase per selected timestep
        L_adv = L.copy()
        for a in range(adversarial_budget):
            idx = sorted_indices[a]
            L_adv[idx] *= 1.0 + 1e-1 * np.sign(peak_cost_derivative[idx])


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

    peak_cost = peak_shaving_cost_from_results(p_grid)
    return peak_cost