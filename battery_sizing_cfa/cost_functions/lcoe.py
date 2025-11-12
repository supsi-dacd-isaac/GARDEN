import numpy as np

def lcoe_from_results(
    L,
    PV_base,
    price,
    export_price,
    Delta_t,
    x_pv,
    E_bat_kWh,
    P_bat_max_kW,
    c_PV_kw,
    c_bat_E_kWh,
    c_bat_P_kw,
    peak_tariff_per_MW_period,
    peak_period_steps,
    discount_rate,
    lifetime_years,
    replicate_periods,
    P_net_kW=None,
    period_peaks_MW=None, # This should be in MW to match the tariff unit
    peak_tariff_profile=None, # override peak_tariff_per_MW_period
    installation_fixed_costs=200,
    **kwargs
):
    """
    Calculates LCOE based on system specifications and operational data.
    Assumes operational series (P_net_kW, period_peaks_MW) are provided
    e.g., from the optimization result or a simulation.
    """
    N = len(L)
    PV_base_peak = float(np.max(PV_base))
    # Check if PV_base has any positive values before setting PV_base_peak for capex
    if PV_base_peak <= 0 and any(PV_base > 0) and c_PV_kw > 0:
        # This condition should ideally not be met if PV_base has positive values,
        # but it's here for robustness and consistency with the optimization function.
        if np.max(PV_base) <= 0 and c_PV_kw > 0:
             raise ValueError("PV_base_peak must be > 0 if PV_base has positive values and c_PV_kw > 0")
        else:
             # If PV_base is all zero or negative, set peak to 0 for capex calculation
             PV_base_peak = 0.0
    elif PV_base_peak <= 0:
         # If PV_base peak is non-positive, set it to 0 for capex calculation
         PV_base_peak = 0.0


    # Annualization & scaling
    # To match the optimization's definition of the denominator (total annual load served),
    # we use the original load profile L and the replicate_periods factor.
    energy_horizon_kWh = float(np.sum(L) * Delta_t)
    energy_annual_kWh = replicate_periods * energy_horizon_kWh

    # Capital Recovery Factor
    r, n = discount_rate, lifetime_years
    # Avoid division by zero if lifetime_years is 0 or discount_rate is 0 and lifetime_years > 0
    if n == 0:
        crf = 1.0 if r > 0 else 0.0 # Annual cost is total cost if lifetime is 0 (not realistic but for completeness)
    elif r == 0:
        crf = 1.0 / n
    else:
        crf = (r * (1 + r)**n) / ((1 + r)**n - 1)


    # Capex (annualized)
    capex = (c_PV_kw * x_pv * PV_base_peak # kW * $/kW
             + c_bat_E_kWh * E_bat_kWh      # kWh * $/kWh
             + c_bat_P_kw  * P_bat_max_kW) # kW * $/kW

    if E_bat_kWh>0 or x_pv>0:
        capex += installation_fixed_costs

    capex_annual = crf * capex

    # Opex (annualized)
    if P_net_kW is None:
        raise ValueError("P_net_kW must be provided.")

    # Separate imports and exports from net load
    P_imp_kW = np.maximum(0, P_net_kW)
    P_exp_kW = np.maximum(0, -P_net_kW)

    price_import = np.asarray(price, float) # $/MWh
    export_price = np.asarray(export_price, float)

    # Convert kW to MW for price calculation ($/MWh * MWh)
    opex_energy = sum(price_import[t] * (P_imp_kW[t]/1000.0) * Delta_t for t in range(N)) \
            - sum(export_price[t] * (P_exp_kW[t]/1000.0) * Delta_t for t in range(N))

    # Curtailment cost - Removed as per user request
    # curt_cost    = curtail_penalty_per_MWh * sum((P_curt_kW[t]/1000.0) * Delta_t for t in range(N))
    curt_cost = 0.0 # Explicitly set to 0 to match optimization

    peak_cost    = 0.0
    use_period_peaks = (peak_period_steps is not None) and (peak_period_steps > 0) and (
        (period_peaks_MW is not None and len(period_peaks_MW) > 0) or (peak_tariff_per_MW_period > 0.0)
    )
    if use_period_peaks:
        P = int(np.ceil(N / peak_period_steps))
        if period_peaks_MW is None:
             raise ValueError("period_peaks_MW must be provided if peak_period_steps > 0.")

        if peak_tariff_profile is not None:
             peak_tariffs = np.asarray(peak_tariff_profile, float)
             if len(peak_tariffs) != P:
                 raise ValueError(f"peak_tariff_profile length {len(peak_tariffs)} != periods {P}")
        else:
            peak_tariffs = np.full(P, float(peak_tariff_per_MW_period), dtype=float)

        # Peak tariff is in $/MW, peak is in MW, so multiply directly
        # The optimization was converting kW peak to MW for the tariff,
        # and period_peaks_MW from the optimization results is already in MW.
        peak_cost = sum(peak_tariffs[p] * period_peaks_MW[p] for p in range(np.minimum(P, len(period_peaks_MW))))


    # replicate operating costs across the year
    opex_annual = replicate_periods * (opex_energy + curt_cost + peak_cost)

    total_annual_cost = capex_annual + opex_annual

    # Objective is LCOE in $/MWh, so total cost in $ / energy in MWh
    # Use energy_annual_kWh for the denominator as per the optimization's definition
    if energy_annual_kWh <= 0:
        lcoe = float('inf') # Avoid division by zero
    else:
        lcoe = total_annual_cost / (energy_annual_kWh / 1000.0)

    return lcoe

