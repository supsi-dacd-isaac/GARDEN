import cvxpy as cp
import numpy as np
from tqdm import tqdm

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

    prob = peak_shaving_solver(T, eta_ch, eta_dis, E_max, E_min, P_max)
    # Set parameters
    prob.parameters()[0].value = p_load - p_pv
    prob.parameters()[1].value = E_init
    # Solve
    prob.solve(solver=cp.GUROBI, verbose=False, warm_start=True)
    p_batt = prob.variables()[0].value - prob.variables()[1].value
    E = prob.variables()[1].value
    return {
        "p_batt": p_batt,
        "E": E,
        "objective": prob.value,
        "status": prob.status,
    }


def peak_shaving_solver(T, eta_ch=0.95, eta_dis=0.9, E_max=10.0, E_min=0.0, P_max=5.0):
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


    # Decision variables
    p_pos = cp.Variable(T, nonneg=True, name='p_pos')    # charging power
    p_neg = cp.Variable(T, nonneg=True, name='p_neg')    # discharging power
    E = cp.Variable(T+1, name='E')                     # state of charge

    p_load = cp.Parameter(T, name='p_load')  # net load profile
    E_init = cp.Parameter(nonneg=True, name='E_init')


    # Energy balance constraints
    #constraints = [E[0] == E_init + eta_ch * p_pos[0] - (1/eta_dis) * p_neg[0]]
    constraints = [E[0] == E_init]
    for t in range(1, T+1):
        constraints += [
            E[t] == E[t-1] + eta_ch * p_pos[t-1] - (1/eta_dis) * p_neg[t-1]
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
    p_batt = p_pos - p_neg
    grid_power = p_load + p_batt
    objective = cp.Minimize(cp.sum_squares(grid_power)/T)

    # Problem
    prob = cp.Problem(objective, constraints)
    params = {
        'p_load': p_load,
        'E_init': E_init,
    }
    return prob, params

def dyn_prices_solver(T, eta_ch=0.95, eta_dis=0.90,
                      E_max=10.0, E_min=0.0, P_max=5.0,
                      require_terminal_equal=False):
    """
    Dynamic tariff LP:
      min  sum_t (buy_t * g_pos_t - sell_t * g_neg_t)
      s.t. g_pos - g_neg = p_load + p_pos - p_neg
           E[0] = E_init
           E[1:] = E[:-1] + eta_ch*p_pos - (1/eta_dis)*p_neg
           E_min <= E <= E_max
           0 <= p_pos, p_neg <= P_max
           0 <= g_pos, g_neg
    If require_terminal_equal=True, enforce E[T] = E_init (cycle).
    """
    # Decision variables
    p_pos = cp.Variable(T, nonneg=True, name="p_pos")     # charge power
    p_neg = cp.Variable(T, nonneg=True, name="p_neg")     # discharge power
    g_pos = cp.Variable(T, nonneg=True, name="g_pos")     # grid import
    g_neg = cp.Variable(T, nonneg=True, name="g_neg")     # grid export (sell)
    E     = cp.Variable(T+1, name="E")                    # energy [kWh]

    # Parameters
    buying  = cp.Parameter(T, nonneg=True, name="buying_price")
    selling = cp.Parameter(T, nonneg=True, name="selling_price")
    p_load  = cp.Parameter(T, name="p_load")              # can be +/- (net demand)
    E_init  = cp.Parameter(nonneg=True, name="E_init")

    # Constraints (vectorized)
    cons = [
        E[0] == E_init,
        E[1:] == E[:-1] + eta_ch * p_pos - (1.0/eta_dis) * p_neg,
        E_min <= E, E <= E_max,
        p_pos <= P_max,
        p_neg <= P_max,
        # grid power split
        g_pos - g_neg == p_load + p_pos - p_neg,
    ]
    if require_terminal_equal:
        cons += [E[-1] == E_init]

    # Linear objective: buy imports, subtract sell revenue on exports
    obj = cp.Minimize(buying @ g_pos - selling @ g_neg + 0.001*cp.sum_squares(g_pos))
    prob = cp.Problem(obj, cons)

    params = dict(buying_price=buying, selling_price=selling, p_load=p_load, E_init=E_init)
    vars   = dict(p_pos=p_pos, p_neg=p_neg, g_pos=g_pos, g_neg=g_neg, E=E)
    return prob, params, vars


def mpc_peak_shaving(p_load_forecasts, eta_ch=0.95, eta_dis=0.9, E_max=10.0, E_min=0.0, P_max=5.0, E_init=5.0):
    """
    Model Predictive Control for peak shaving.

    Parameters:
        p_load_forecasts: matrtix-like, T x H, load profile
        eta_ch: float, charging efficiency
        eta_dis: float, discharging efficiency
        E_max: float, maximum energy capacity
        E_min: float, minimum energy capacity
        P_max: float, maximum charge/discharge power
        E_init: float, initial state of charge
        horizon: int, prediction horizon
    Returns:
        dict with keys:
            "p_batt": battery power profile
            "E": state of charge profile
            "objective": total objective value
    """


    T, H = p_load_forecasts.shape
    p_batt_profile = np.zeros(T)
    E_profile = np.zeros(T)
    E_current = E_init
    total_objective = 0.0

    solver, params = peak_shaving_solver(H, eta_ch, eta_dis, E_max, E_min, P_max)

    for t in tqdm(range(T)):
        # Extract the load and PV profiles for the horizon
        p_load_horizon = p_load_forecasts[t]
        p_pv_horizon = np.zeros(H)  # Assuming no PV for peak shaving

        #set solver parameters
        params['p_load'].value = p_load_horizon - p_pv_horizon
        params['E_init'].value = E_current


        # Solve
        result = solver.solve(solver=cp.GUROBI, verbose=False, warm_start=True)
        p_batt_pos = solver.variables()[0].value
        p_batt_neg = solver.variables()[1].value
        p_batt = p_batt_pos - p_batt_neg

        E = solver.variables()[2].value

        # Apply the first control action
        p_batt_profile[t] = np.copy(p_batt[0])
        E_current = E[1]  # Update state of charge
        E_profile[t] = np.copy(E_current)
        total_objective += result  # Accumulate objective

    return {
        "p_grid": p_load_forecasts[:,0] + p_batt_profile,
        "p_batt": p_batt_profile,
        "E": E_profile,
        "objective": total_objective,
    }

def mpc_dyn_tariff(p_load_forecasts, p_sell, p_buy, eta_ch=0.95, eta_dis=0.9, E_max=10.0, E_min=0.0, P_max=5.0, E_init=5.0):
    """
    Model Predictive Control for dynamic price.

    Parameters:
        p_load_forecasts: matrtix-like, T x H, load profile
        p_sell: list like, T, selling price profile, known in advance
        p_buy: list like, T, buying price profile, known in advance
        eta_ch: float, charging efficiency
        eta_dis: float, discharging efficiency
        E_max: float, maximum energy capacity
        E_min: float, minimum energy capacity
        P_max: float, maximum charge/discharge power
        E_init: float, initial state of charge
        horizon: int, prediction horizon
    Returns:
        dict with keys:
            "p_batt": battery power profile
            "E": state of charge profile
            "objective": total objective value
    """

    T, H = p_load_forecasts.shape
    p_batt_profile = np.zeros(T)
    E_profile = np.zeros(T)
    E_current = E_init
    total_objective = 0.0

    solver, params, vars = dyn_prices_solver(H, eta_ch, eta_dis, E_max, E_min, P_max)

    # augment psell and pbuy to be at least length T+H (zero holding at end)
    if len(p_sell) < T+H:
        p_sell = np.concatenate([p_sell, p_sell[-1]*np.ones(T+H - len(p_sell))])
    if len(p_buy) < T+H:
        p_buy = np.concatenate([p_buy, p_buy[-1]*np.ones(T+H - len(p_buy))])

    for t in tqdm(range(T)):
        # Extract the load and PV profiles for the horizon
        p_load_horizon = p_load_forecasts[t]

        # Extract price profiles for the horizon
        selling_price_horizon = p_sell[t:t+H]
        buying_price_horizon = p_buy[t:t+H]

        #set solver parameters
        params['selling_price'].value = selling_price_horizon
        params['buying_price'].value = buying_price_horizon
        params['p_load'].value = p_load_horizon
        params['E_init'].value = E_current

        # Solve
        result = solver.solve(solver=cp.GUROBI, verbose=False, warm_start=True)
        p_batt_pos = vars['p_pos'].value
        p_batt_neg = vars['p_neg'].value
        p_batt = p_batt_pos - p_batt_neg

        E = solver.variables()[2].value

        # Apply the first control action
        p_batt_profile[t] = np.copy(p_batt[0])
        E_current = E[1]  # Update state of charge
        E_profile[t] = np.copy(E_current)
        total_objective += result  # Accumulate objective

    return {
        "p_grid": p_load_forecasts[:,0] + p_batt_profile,
        "p_batt": p_batt_profile,
        "E": E_profile,
        "objective": total_objective,
    }