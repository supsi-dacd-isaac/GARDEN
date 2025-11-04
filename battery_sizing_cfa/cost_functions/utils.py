import numpy as np
from battery_sizing_cfa.cost_functions.peak_shaving import loss_batched_with_h, rbc_thresholds_batched_parallel
import pandas as pd

def build_block_basis(T, block_len, stride=None, normalize=True):
    """
    Return U: (T, m) block basis with ones over blocks, zeros elsewhere.
    - block_len: window length in steps (e.g., 24 for daily if hourly and 24h/day)
    - stride: shift between consecutive blocks (default=block_len -> non-overlapping)
    """
    if stride is None:
        stride = block_len
    cols = []
    for start in range(0, T, stride):
        end = min(start + block_len, T)
        if end <= start:
            break
        u = np.zeros(T, dtype=float)
        u[start:end] = 1.0
        if normalize:
            u[start:end] /= np.sqrt(end - start)
        cols.append(u)
        if end == T:
            break
    U = np.stack(cols, axis=1)  # (T, m)
    return U

def block_nes_grad(L, h, lower_thr, upper_thr, specs,
                   U, sigma=0.3, day_len=24, tau=0.5):
    """
    L: (T,) base profile
    U: (T, m) block directions
    returns g: (T,) gradient-like score
    """
    m = U.shape[1]
    X_plus  = L[np.newaxis, :] + sigma * U.T   # (m, T)
    X_minus = L[np.newaxis, :] - sigma * U.T   # (m, T)
    vals = loss_batched_with_h(
        np.vstack([X_plus, X_minus]),
        h,
        lower_thr, upper_thr, specs,
        tau=tau
    )  # (2m,)
    f_plus, f_minus = vals[:m], vals[m:]
    g = (U @ (f_plus - f_minus)) / (2.0 * sigma * m)  # (T,)
    return g


def block_nes_grad_with_h_parallel(
    L, h, lower_thr, upper_thr, specs, U, sigma=0.3, tau=0.5, day_len=None
):
    """
    L: (T,) base profile
    h: (T,) hour-of-day vector used for daily max binning
    U: (T, m) block directions (columns)
    Returns: grads (T,)
    """
    # --- dimensions
    T = L.size
    assert U.shape[0] == T, "U must have shape (T, m)"
    m = U.shape[1]

    # --- build batch of perturbed profiles (m, T), C-contiguous float64
    X_plus  = (L[np.newaxis, :] + sigma * U.T).astype(np.float64, copy=False)
    X_minus = (L[np.newaxis, :] - sigma * U.T).astype(np.float64, copy=False)
    X_batch = np.ascontiguousarray(np.vstack([X_plus, X_minus]))  # (2m, T)

    # --- expand thresholds to (B, T) with B = m (not T!)
    lowerT, upperT = _expand_thresholds(lower_thr, upper_thr, m, T)
    # make two copies because we have two batches (plus/minus) stacked
    lower2 = np.ascontiguousarray(np.vstack([lowerT, lowerT]))
    upper2 = np.ascontiguousarray(np.vstack([upperT, upperT]))

    # --- run controller in parallel (once) on the whole batch
    socB, pbatB, pgridB = rbc_thresholds_batched_parallel(
        X_batch,
        capacity_kwh=specs.get('capacity_kwh', 100.0),
        soc_start=specs.get('soc_start', 0.5),
        soc_min=specs.get('soc_min', 0.10),
        soc_max=specs.get('soc_max', 0.99),
        p_charge_max=specs.get('p_charge_max', 50.0),
        p_discharge_max=specs.get('p_discharge_max', 50.0),
        eta_ch=specs.get('eta_ch', 0.95),
        eta_dis=specs.get('eta_dis', 0.95),
        dt_hours=specs.get('dt_hours', 1.0),
        lowerT=lower2,
        upperT=upper2,
        do_not_charge_from_grid=specs.get('do_not_charge_from_grid', False)
    )

    # --- daily max loss (hard or soft) on the resulting grid power
    pg_pos = np.clip(pgridB, 0.0, None)  # (2m, T)

    # bin days from h
    if day_len is None:
        # build by detecting wrap in h
        day_id = np.cumsum(np.diff(h, prepend=h[0]) < 0)
        n_days = int(day_id.max() + 1)
    else:
        # if you pass a known day_len (e.g., 24/96), use modulo
        day_id = (np.arange(T) % int(day_len))
        # make consecutive-day ids:
        day_id = np.cumsum(np.diff(day_id, prepend=day_id[0]) < 0)
        n_days = int(day_id.max() + 1)

    # soft daily max (better gradients than hard max)
    def soft_daily_max_sum_1d(arr, day_id, n_days, tau=0.5):
        # arr: (T,)
        # stabilize per-day with hard max
        out = -1.0
        # compute per-day max
        M = np.full(n_days, -np.inf)
        np.maximum.at(M, day_id, arr)
        z = np.exp((arr - M[day_id]) / tau)
        sumexp = np.bincount(day_id, weights=z, minlength=n_days)
        return float(np.sum(tau * np.log(sumexp) + M))

    # evaluate losses for all 2m profiles
    vals = np.empty(2 * m, dtype=np.float64)
    for i in range(2 * m):
        vals[i] = soft_daily_max_sum_1d(pg_pos[i], day_id, n_days, tau=tau)
    f_plus, f_minus = vals[:m], vals[m:]

    # NES aggregation with the block basis
    grads = (U @ (f_plus - f_minus)) / (2.0 * sigma * m)  # (T,)
    return grads

# ---------------------------
# 4) SPSA (dimension-free, 2 evals)
# ---------------------------
def spsa_grad(L, h, lower_thr, upper_thr, specs,
              c=0.3, day_len=24, tau=0.5):
    d = np.where(np.random.rand(*L.shape) < 0.5, -1.0, 1.0)
    fp = loss_batched_with_h(
        (L + c*d)[None, :], h, lower_thr, upper_thr, specs, tau=tau
    )[0]
    fm = loss_batched_with_h(
        (L - c*d)[None, :], h, lower_thr, upper_thr, specs, tau=tau
    )[0]
    return ((fp - fm) / (2.0*c)) * d



def _expand_thresholds(lower_threshold, upper_threshold, B, T):
    """Return (lowerT, upperT) shaped (B, T) for numba-friendly use."""
    # lower
    if np.ndim(lower_threshold) == 0 or (np.ndim(lower_threshold) == 1 and lower_threshold.size == 1):
        lowerT = np.full((B, T), float(np.ravel(lower_threshold)[0]))
    elif np.ndim(lower_threshold) == 1 and lower_threshold.size == T:
        lowerT = np.tile(lower_threshold.astype(np.float64), (B, 1))
    elif np.ndim(lower_threshold) == 2 and lower_threshold.shape == (B, T):
        lowerT = lower_threshold.astype(np.float64)
    else:
        raise ValueError("lower_threshold must be scalar, (T,), or (B,T)")

    # upper
    if np.ndim(upper_threshold) == 0 or (np.ndim(upper_threshold) == 1 and upper_threshold.size == 1):
        upperT = np.full((B, T), float(np.ravel(upper_threshold)[0]))
    elif np.ndim(upper_threshold) == 1 and upper_threshold.size == T:
        upperT = np.tile(upper_threshold.astype(np.float64), (B, 1))
    elif np.ndim(upper_threshold) == 2 and upper_threshold.shape == (B, T):
        upperT = upper_threshold.astype(np.float64)
    else:
        raise ValueError("upper_threshold must be scalar, (T,), or (B,T)")

    return lowerT, upperT


def cvar_from_daily_losses(losses, alpha):
    """losses: (D,) per-day losses; returns empirical CVaR_alpha."""
    if alpha == 1.0:
        return np.mean(losses)
    D = losses.size
    k_real = (1.0 - alpha) * D
    order = np.partition(losses, D - int(np.ceil(k_real)))  # O(D)
    # indices of top ceil(k_real) elements
    k_hi = int(np.ceil(k_real))
    tail = np.sort(order[-k_hi:])  # small sort
    if np.isclose(k_real, k_hi):
        return float(tail.mean())
    # interpolate between top floor(k_real) and the next one
    k_lo = k_hi - 1
    w = k_real - k_lo
    if k_lo == 0:  # degenerate: CVaR equals the max
        return float(tail[-1])
    return float((tail[-k_lo:].sum() + w * tail[-(k_lo + 1)]) / k_real)

# --- CVaR worst-case weights (risk-envelope). Returns w aligned with original order.
def cvar_worst_weights(losses, alpha):
    D = losses.size
    k_real = (1.0 - alpha) * D
    w = np.zeros(D, dtype=float)
    if k_real <= 0:               # degenerate alpha→1: all weight on the single worst
        w[np.argmax(losses)] = 1.0
        return w

    cap = 1.0 / ((1.0 - alpha) * D)   # per-sample weight cap
    order_desc = np.argsort(losses)[::-1]  # indices from largest to smallest

    k_hi = int(np.ceil(k_real))
    k_lo = max(0, k_hi - 1)

    # full-cap weights on the top floor(k_real)
    for j in range(k_lo):
        w[order_desc[j]] = cap

    # fractional weight on the next one (handles non-integer k_real)
    if k_lo < k_hi:
        frac = (k_real - k_lo) * cap   # in [0, cap]
        w[order_desc[k_lo]] = frac

    # (optional) tiny numerical tidy
    s = w.sum()
    if s != 0.0 and not np.isclose(s, 1.0):
        w /= s
    return w

# --- CVaR as weighted mean (exactly equals CVaR_alpha by theory)
def cvar_weighted_mean(losses, alpha):
    w = cvar_worst_weights(losses, alpha)
    return float(np.dot(w, losses)), w

def frac_rolling_quantile(x, W_star, q, min_periods=1):
    n  = int(np.floor(W_star))
    d  = float(W_star - n)
    # guard for very small windows
    n1 = max(1, n)
    n2 = max(1, n+1)

    s = pd.Series(x)
    q1 = s.rolling(window=n1, min_periods=min_periods).quantile(q).to_numpy()
    q2 = s.rolling(window=n2, min_periods=min_periods).quantile(q).to_numpy()
    return (1.0 - d) * q1 + d * q2