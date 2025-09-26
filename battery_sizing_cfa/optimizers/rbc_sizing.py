import numpy as np
from battery_sizing_cfa.cost_functions.lcoe import lcoe_from_results
from battery_sizing_cfa.optimizers.parametric_rbc import rbc


def rbc_opt_fun(sampled_pars, L, PV_base, price, export_price, specs, **kwargs):

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
    soc_max=specs['soc_max'])

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
