import numpy as np
from numba import njit
from numba import prange

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
    dt_hours=1.0,           # time step [h]
    noise_level=0
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
        soc_min_price = soc_min_price_fun[t]*(1 + np.random.uniform(-1, 1)*noise_level)
        p_threshold = p_threshold_fun[t]*(1 + np.random.uniform(-1, 1)*noise_level)

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


@njit
def rbc_thresholds(
    p_tot,                 # array [T], + = net consumption, - = net production (surplus)
    price_high,            # array [T] of 0/1 (unused here, kept for signature compatibility)
    capacity_kwh=100.0,    # usable capacity [kWh]
    soc_start=0.5,         # initial SOC in [0,1]
    soc_min=0.10,          # hard minimum SOC (never go below)
    soc_max=0.99,          # hard maximum SOC (never exceed)
    p_charge_max=50.0,     # max charge power [kW]
    p_discharge_max=50.0,  # max discharge power [kW]
    eta_ch=0.95,           # charging efficiency
    eta_dis=0.95,          # discharging efficiency
    dt_hours=1.0,          # time step [h]
    noise_level=0,
    lower_threshold=np.array([0.0]),
    upper_threshold=np.array([1.0]),
    do_not_charge_from_grid=False
):
    T = p_tot.shape[0]
    soc = np.empty(T, dtype=np.float64)
    p_batt = np.zeros(T, dtype=np.float64)
    p_grid = np.zeros(T, dtype=np.float64)

    # Internal energy state [kWh]
    E = soc_start * capacity_kwh
    E_min_hard = soc_min * capacity_kwh
    E_max = soc_max * capacity_kwh

    # Handle zero/negative time step
    if dt_hours <= 0:
        soc[:] = soc_start
        p_batt[:] = 0.0
        p_grid[:] = p_tot
        return soc, p_batt, p_grid

    # Expand thresholds if scalar-like arrays provided
    if lower_threshold.shape[0] == 1:
        lower_threshold = lower_threshold * np.ones(T)
    if upper_threshold.shape[0] == 1:
        upper_threshold = upper_threshold * np.ones(T)
    alpha_ch, alpha_dis = 0.2, 0.2  # perturbation factors
    for t in range(T):
        # optional perturbation

        lt = lower_threshold[t] #* (1 + alpha_ch * (0.5 - soc[t - 1]))
        ut = upper_threshold[t] #* (1 + alpha_dis * (0.5 - soc[t - 1]))

        pt = p_tot[t]

        # Decide action: discharge if pt > ut, charge if pt < lt, else idle
        if pt > ut:
            # Discharge amount = amount above upper threshold
            desired = pt - ut
            if desired < 0.0:
                desired = 0.0

            # energy-limited discharge (don't go below hard soc_min)
            if E > E_min_hard:
                p_by_energy = (E - E_min_hard) * eta_dis / dt_hours
            else:
                p_by_energy = 0.0

            P = desired
            if P > p_discharge_max:
                P = p_discharge_max
            if P > p_by_energy:
                P = p_by_energy

            # apply discharge (battery supplies load -> p_batt negative)
            p_batt[t] = -P
            E -= (P / eta_dis) * dt_hours

        elif pt < lt:
            # Charge: absorb surplus (pt negative -> surplus = -pt)
            if do_not_charge_from_grid and pt < 0.0:
                surplus = -pt if pt < 0.0 else 0.0
            else:
                surplus = np.abs(pt-lt)
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

        else:
            p_batt[t] = 0.0

        # enforce hard energy bounds
        if E < E_min_hard:
            E = E_min_hard
        if E > E_max:
            E = E_max

        # SOC (handle zero capacity)
        if capacity_kwh > 0.0:
            soc[t] = E / capacity_kwh
        else:
            soc[t] = soc_start

        p_grid[t] = pt + p_batt[t]

    return soc, p_batt, p_grid



@njit
def _get_thr(lower_threshold, upper_threshold, b, t, T):
    # Returns (lt, ut) for batch b, time t, handling scalar, (T,), or (B,T)
    if lower_threshold.ndim == 1:
        if lower_threshold.size == 1:
            lt = lower_threshold[0]
        else:  # (T,)
            lt = lower_threshold[t]
    else:  # (B,T)
        lt = lower_threshold[b, t]

    if upper_threshold.ndim == 1:
        if upper_threshold.size == 1:
            ut = upper_threshold[0]
        else:  # (T,)
            ut = upper_threshold[t]
    else:  # (B,T)
        ut = upper_threshold[b, t]
    return lt, ut


@njit(parallel=True, fastmath=True)
def rbc_thresholds_batched_parallel(
    p_tot,                 # (B, T)
    capacity_kwh=100.0,
    soc_start=0.5,
    soc_min=0.10,
    soc_max=0.99,
    p_charge_max=50.0,
    p_discharge_max=50.0,
    eta_ch=0.95,
    eta_dis=0.95,
    dt_hours=1.0,
    lowerT=None,          # (B, T) pre-expanded
    upperT=None,          # (B, T) pre-expanded
    do_not_charge_from_grid=False
):
    B, T = p_tot.shape
    soc   = np.empty((B, T), dtype=np.float64)
    p_bat = np.zeros((B, T), dtype=np.float64)
    p_grid= np.zeros((B, T), dtype=np.float64)

    E_min_hard = soc_min * capacity_kwh
    E_max      = soc_max * capacity_kwh

    if dt_hours <= 0.0:
        for b in prange(B):
            for t in range(T):
                soc[b, t] = soc_start
                p_bat[b, t]= 0.0
                p_grid[b,t]= p_tot[b,t]
        return soc, p_bat, p_grid

    # Parallelize over series
    for b in prange(B):
        E = soc_start * capacity_kwh
        for t in range(T):
            lt = lowerT[b, t]
            ut = upperT[b, t]
            pt = p_tot[b, t]

            if pt > ut:
                desired = pt - ut
                if desired < 0.0:
                    desired = 0.0

                if E > E_min_hard:
                    p_by_energy = (E - E_min_hard) * eta_dis / dt_hours
                else:
                    p_by_energy = 0.0

                P = desired
                if P > p_discharge_max:
                    P = p_discharge_max
                if P > p_by_energy:
                    P = p_by_energy

                p_bat[b, t] = -P
                E -= (P / eta_dis) * dt_hours

            elif pt < lt:
                if do_not_charge_from_grid and pt < 0.0:
                    surplus = -pt
                else:
                    diff = lt - pt
                    surplus = diff if diff > 0.0 else 0.0

                if E < E_max:
                    p_by_room = (E_max - E) / (eta_ch * dt_hours)
                else:
                    p_by_room = 0.0

                P = surplus
                if P > p_charge_max:
                    P = p_charge_max
                if P > p_by_room:
                    P = p_by_room

                p_bat[b, t] = P
                E += (P * eta_ch) * dt_hours

            else:
                p_bat[b, t] = 0.0

            # clamp energy and write outputs
            if E < E_min_hard:
                E = E_min_hard
            if E > E_max:
                E = E_max

            if capacity_kwh > 0.0:
                soc[b, t] = E / capacity_kwh
            else:
                soc[b, t] = soc_start

            p_grid[b, t] = pt + p_bat[b, t]

    return soc, p_bat, p_grid