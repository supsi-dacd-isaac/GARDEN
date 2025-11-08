import numpy as np
from battery_sizing_cfa.optimizers.parametric_rbc import rbc_thresholds_batched_parallel

def peak_shaving_cost_from_results(L):
    return np.mean(L**2)

def daily_maxima(L, h=None):
    #day_id = np.cumsum(np.diff(h, prepend=h[0]) < 0)  # or your own day labels
    #out = np.full(day_id.max() + 1, -np.inf)
    #np.maximum.at(out, day_id, L)
    day_id, _ = make_day_id(h)
    out = daily_maxima_1d(L, day_id, n_days=int(day_id.max() + 1))
    return out

def day_max_cost_from_results(L, h=None):
    daily_max = daily_maxima(L, h)
    return np.nanmean(daily_max)


def make_day_id(h):
    """
    h: (T,) array of hour-of-day (e.g., 0..23 or 0,0.25,...,23.75)
    Returns:
      day_id: (T,) int array labeling each timestamp's day (0..n_days-1)
      n_days: number of days
    """
    # new day starts when hour decreases (wraps after 23 -> 0)
    day_id = np.cumsum(np.diff(h, prepend=h[0]) < 0)
    return day_id, int(day_id.max() + 1)

def daily_maxima_1d(L, day_id, n_days):
    """
    L: (T,) series
    day_id: (T,) labels in [0..n_days-1]
    """
    out = np.full(n_days, -np.inf, dtype=float)
    np.maximum.at(out, day_id, L)
    return out

def daily_max_sum_1d(L, day_id, n_days):
    return np.sum(daily_maxima_1d(L, day_id, n_days))

def soft_daily_max_sum_1d(L, day_id, n_days, tau=0.5):
    # per-day hard max for stabilization
    M = daily_maxima_1d(L, day_id, n_days)                  # (n_days,)
    # sum exp((L - M_day)/tau) per day via bincount
    z = np.exp((L - M[day_id]) / tau)                       # (T,)
    sumexp = np.bincount(day_id, weights=z, minlength=n_days)  # (n_days,)
    return float(np.sum(tau * np.log(sumexp) + M))


def daily_max_sum_batched(LB, day_id, n_days):
    """
    LB: (B,T)
    Returns (B,)
    """
    B, T = LB.shape
    out = np.empty(B, dtype=float)
    for b in range(B):
        out[b] = daily_max_sum_1d(LB[b], day_id, n_days)
    return out

def soft_daily_max_sum_batched(LB, day_id, n_days, tau=0.5):
    B, T = LB.shape
    out = np.empty(B, dtype=float)
    for b in range(B):
        out[b] = soft_daily_max_sum_1d(LB[b], day_id, n_days, tau=tau)
    return out

def loss_batched_with_h(LB, h, lower_thr, upper_thr, specs, use_soft=True, tau=0.5):
    """
    LB: (B,T) batch of net loads
    h : (T,) hours-of-day for binning daily maxima
    """
    day_id, n_days = make_day_id(h)

    # Run your numba-jitted batched controller
    _, _, p_grid = rbc_thresholds_batched_parallel(
        LB,
        capacity_kwh=specs.get('capacity_kwh', 100.0),
        soc_start=specs.get('soc_start', 0.5),
        soc_min=specs.get('soc_min', 0.10),
        soc_max=specs.get('soc_max', 0.99),
        p_charge_max=specs.get('p_charge_max', 50.0),
        p_discharge_max=specs.get('p_discharge_max', 50.0),
        eta_ch=specs.get('eta_ch', 0.95),
        eta_dis=specs.get('eta_dis', 0.95),
        dt_hours=specs.get('dt_hours', 1.0),
        lowerT=lower_thr,   # scalar (1,), (T,) or (B,T)
        upperT=upper_thr,   # scalar (1,), (T,) or (B,T)
        do_not_charge_from_grid=specs.get('do_not_charge_from_grid', False)
    )

    # Only penalize positive grid import for peak shaving (optional)
    pg_pos = np.clip(p_grid, 0.0, None)  # (B,T)

    if use_soft:
        return soft_daily_max_sum_batched(pg_pos, day_id, n_days, tau=tau)  # (B,)
    else:
        return daily_max_sum_batched(pg_pos, day_id, n_days)