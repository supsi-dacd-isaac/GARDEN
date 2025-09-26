import numpy as np
from numba import njit


@njit
def rbc(
    p_tot,                 # array [T], + = net consumption, - = net production (surplus)
    price_high,            # array [T] of 0/1 (1 = high price)
    capacity_kwh=100.0,    # usable capacity [kWh]
    soc_start=0.5,              # initial SOC in [0,1]
    soc_min=0.10,          # hard minimum SOC (never go below)
    soc_max=0.99,          # hard maximum SOC (never exceed)
    soc_min_price_fun=np.array([0.20]),    # price-discharge minimum SOC (stop price-driven discharge here)
    p_charge_max=50.0,     # max charge power [kW]
    p_discharge_max=50.0,  # max discharge power [kW]
    eta_ch=0.95,           # charging efficiency
    eta_dis=0.95,          # discharging efficiency
    p_threshold_fun=np.array([0.0]),       # rule (3): if p_tot > p_threshold, try discharging
    dt_hours=1.0           # time step [h]
):
    """
    Implements priority rules:
      (1) If price is high AND p_tot>0 -> discharge (stop at soc_min_price).
      (2) Else if p_tot<0 -> charge (stop at soc_max).
      (3) Else if p_tot>p_threshold -> discharge (stop at soc_min).
      Else idle.

    Returns:
      soc    : array [T] state of charge in [0,1]
      p_batt : array [T] battery power (+ discharge to load, - charge from grid/surplus)
      p_grid : array [T] residual grid power after battery action
    """
    T = p_tot.shape[0]
    soc = np.empty(T, dtype=np.float64)
    p_batt = np.zeros(T, dtype=np.float64)
    p_grid = np.zeros(T, dtype=np.float64)



    # Internal energy state [kWh]
    E = soc_start * capacity_kwh
    E_min_hard = soc_min * capacity_kwh
    E_max = soc_max * capacity_kwh

    # Handle zero time step
    if dt_hours <= 0:
        # If time step is zero or negative, no change in SOC can occur
        # and no power can be transferred. Return initial state for all steps.
        soc[:] = soc_start
        p_batt[:] = 0.0
        p_grid[:] = p_tot
        return soc, p_batt, p_grid

    soc_min_price_fun = soc_min_price_fun * np.ones(T) if soc_min_price_fun.shape[0] == 1 else soc_min_price_fun
    p_threshold_fun = p_threshold_fun * np.ones(T) if p_threshold_fun.shape[0] == 1 else p_threshold_fun
    for t in range(T):
        # Ensure consistent limits
        soc_min_price = soc_min_price_fun[t]
        p_threshold = p_threshold_fun[t]

        if soc_min_price < soc_min:
            soc_min_price = soc_min  # never allow a looser floor than soc_min
        E_min_price = soc_min_price * capacity_kwh

        pt = p_tot[t]
        hi = price_high[t] != 0  # bool

        # Decide action (priority 1 → 2 → 3 → else idle)
        action = 0  # 0=idle, 1=price-discharge, 2=charge, 3=threshold-discharge
        if pt > p_threshold:
            action = 3
        elif pt < 0.0:
            action = 2
        elif hi and pt > 0.0:
            action = 1

        if action == 1:
            # Price-driven discharge; stop at soc_min_price
            demand = pt if pt > 0.0 else 0.0
            # Energy-limited power: E should not go below E_min_price
            if E > E_min_price:
                p_by_energy = (E - E_min_price) * eta_dis / dt_hours
            else:
                p_by_energy = 0.0
            P = demand
            if P > p_discharge_max:
                P = p_discharge_max
            if P > p_by_energy:
                P = p_by_energy

            p_batt[t] = -P
            E -= (P / eta_dis) * dt_hours

        elif action == 2:
            # Charge; stop at soc_max
            surplus = -pt if pt < 0.0 else 0.0
            if E < E_max:
                p_by_room = (E_max - E) / (eta_ch * dt_hours)
            else:
                p_by_room = 0.0
            P = surplus
            if P > p_charge_max:
                P = p_charge_max
            if P > p_by_room:
                P = p_by_room

            p_batt[t] = P
            E += (P * eta_ch) * dt_hours

        elif action == 3:
            # Threshold-driven discharge; stop at soc_min (hard)
            demand = pt if pt > 0.0 else 0.0
            p_over_threshold = pt - p_threshold

            if E > E_min_hard:
                p_by_energy = (E - E_min_hard) * eta_dis / dt_hours
            else:
                p_by_energy = 0.0
            P = p_over_threshold
            if P > p_discharge_max:
                P = p_discharge_max
            if P > p_by_energy:
                P = p_by_energy

            p_batt[t] = -P
            E -= (P / eta_dis) * dt_hours

        else:
            p_batt[t] = 0.0  # idle

        # Enforce hard bounds
        if E < E_min_hard:
            E = E_min_hard
        if E > E_max:
            E = E_max

        # Calculate SOC, handle the case where capacity is zero
        if capacity_kwh > 0:
          soc[t] = E / capacity_kwh
        else:
          soc[t] = soc_start # Or some other default/error handling


        p_grid[t] = pt + p_batt[t]

    return soc, p_batt, p_grid