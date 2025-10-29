import cvxpy as cp
import numpy as np

def optimize_peak_shaving(p_load, p_pv,
                                eta_ch=0.95, eta_dis=0.9,
                                E_max=10.0, E_min=0.0,
                                P_max=5.0, E_init=5.0):
    """
    Optimize battery operation for peak shaving using CVXPY + OSQP.

    Objective:
        minimize sum_t (p_load[t] + p_pv[t] + p_batt[t])**2

    Battery dynamics:
        E[t+1] = E[t] + η_ch * p_pos[t] - (1/η_dis) * p_neg[t]
        p_batt[t] = p_pos[t] - p_neg[t]
        0 <= p_pos[t], p_neg[t] <= P_max
        E_min <= E[t] <= E_max
    """
    T = len(p_load)

    # Decision variables
    p_pos = cp.Variable(T, nonneg=True)    # charging power
    p_neg = cp.Variable(T, nonneg=True)    # discharging power
    E = cp.Variable(T)                     # state of charge

    # Battery power (net effect)
    p_batt = p_pos - p_neg

    # Energy balance constraints
    constraints = [E[0] == E_init + eta_ch * p_pos[0] - (1/eta_dis) * p_neg[0]]
    for t in range(1, T):
        constraints += [
            E[t] == E[t-1] + eta_ch * p_pos[t] - (1/eta_dis) * p_neg[t]
        ]

    # Bounds
    constraints += [
        E_min <= E, E <= E_max,
        p_pos <= P_max,
        p_neg <= P_max,
    ]

    # (Optional) cycle constraint
    # constraints += [E[-1] == E_init]

    # Objective: minimize quadratic grid power
    grid_power = p_load - p_pv + p_batt
    objective = cp.Minimize(cp.sum_squares(grid_power)/T)

    # Problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GUROBI, verbose=False)

    return {
        "p_batt": p_batt.value,
        "E": E.value,
        "objective": prob.value,
        "status": prob.status,
    }
