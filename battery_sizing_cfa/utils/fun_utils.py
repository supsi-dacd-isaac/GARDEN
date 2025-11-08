import numpy as np
import pandas as pd
from numba import njit

@njit
def _baseline_from_sorted(x_sorted, target_energy, dt_hours=1.0):
    """
    x_sorted: ascending (n,)
    returns baseline value b s.t. energy_above(b) ~= target_energy
    """
    n = x_sorted.shape[0]
    if n < 2:
        return np.nan

    # suffix sums: suf[k] = sum_{k..n-1} x[k]
    suf = np.empty(n+1)
    suf[n] = 0.0
    for k in range(n-1, -1, -1):
        suf[k] = suf[k+1] + x_sorted[k]

    # energy(i) = (suf[i+1] - (n-1-i) * x[i]) * dt
    # monotone: energy(0) >= ... >= energy(n-1)=0
    def energy_at(i):
        if i >= n-1:
            return 0.0
        return (suf[i+1] - (n-1-i) * x_sorted[i]) * dt_hours

    # boundary checks
    e0 = energy_at(0)
    if e0 < target_energy:
        return x_sorted[0]        # target too large → smallest baseline
    e_last = 0.0
    if e_last > target_energy:
        return x_sorted[-1]       # target tiny → largest baseline

    # binary search for i with energy(i) >= target > energy(i+1)
    lo, hi = 0, n-2
    while lo <= hi:
        mid = (lo + hi) // 2
        e_mid = energy_at(mid)
        if e_mid >= target_energy:
            lo = mid + 1
        else:
            hi = mid - 1

    i = max(0, min(n-2, hi))
    e1 = energy_at(i)
    e2 = energy_at(i+1)
    x1 = x_sorted[i]
    x2 = x_sorted[i+1]

    # linear interpolate baseline between x1 (e1) and x2 (e2)
    if e1 != e2:
        w = (target_energy - e2) / (e1 - e2)
        if w < 0.0: w = 0.0
        if w > 1.0: w = 1.0
        return x2 + w * (x1 - x2)
    else:
        return x1

@njit
def invert_quantile_for_energy_numba(window_vals, target_energy, dt_hours=1.0):
    x = np.sort(window_vals)  # ascending
    return _baseline_from_sorted(x, target_energy, dt_hours)

@njit
def _find_index_equal(arr, val):
    # linear search for value to remove (stable data)
    n = arr.shape[0]
    for i in range(n):
        if arr[i] == val:
            return i
    # fallback if duplicates w/ floating diff: pick closest
    best_i = 0
    best_d = abs(arr[0]-val)
    for i in range(1, n):
        d = abs(arr[i]-val)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i

@njit
def _bisect_right(arr, val):
    # insertion index after any existing equals (ascending)
    lo, hi = 0, arr.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if val < arr[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo

@njit
def rolling_inverted_quantile_np(values, window, energy_trigger, dt_hours=1.0):
    n = values.shape[0]
    out = np.empty(n)
    out[:] = np.nan
    if window <= 1 or n < window:
        return out

    # init sorted window
    buf = np.empty(window)
    for i in range(window):
        buf[i] = values[i]
    buf.sort()  # in-place

    # first output (at t = window-1)
    out[window-1] = _baseline_from_sorted(buf, energy_trigger, dt_hours)

    # slide the window
    for t in range(window, n):
        old_val = values[t - window]
        new_val = values[t]

        # remove old_val: find its index and shift left
        idx_old = _find_index_equal(buf, old_val)
        for k in range(idx_old, window-1):
            buf[k] = buf[k+1]

        # insert new_val at sorted position (bisect right) by shifting right
        ins = _bisect_right(buf[:window-1], new_val)  # search on length window-1
        # shift right to make space
        for k in range(window-1, ins, -1):
            buf[k] = buf[k-1]
        buf[ins] = new_val

        # compute baseline for current window
        out[t] = _baseline_from_sorted(buf, energy_trigger, dt_hours)

    return out


def invert_quantile_for_energy(window_vals, target_energy, dt_hours=1.0):
    x = np.sort(np.asarray(window_vals))
    n = len(x)
    if n < 2:
        return np.nan

    # total energy above each potential quantile
    energy_above = np.zeros(n)
    for i in range(n):
        baseline = x[i]
        energy_above[i] = np.sum(np.maximum(x - baseline, 0)) * dt_hours

    # find where it crosses target_energy
    energy_asc = energy_above[::-1]
    idx = np.searchsorted(energy_asc, target_energy)
    if idx == 0:
        return np.max(window_vals)  # very small energy -> top quantile (100%)
    if idx >= n:
        return np.min(window_vals)  # too large -> bottom quantile (0%)

    q_level = 1.0 - idx / n
    return np.quantile(window_vals, q_level)


def rolling_inverted_quantile(series, window, energy_trigger, dt_hours=1.0):
    return (
        series.rolling(window, min_periods=window)
        .apply(lambda w: invert_quantile_for_energy_numba(w, energy_trigger, dt_hours), raw=False)
    )