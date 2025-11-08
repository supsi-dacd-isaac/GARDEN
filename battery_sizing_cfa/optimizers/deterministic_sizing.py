import pyomo.environ as pyo
import numpy as np
from typing import Optional, Dict, Any

def optimize_lcoe_prescient(
    L: np.ndarray,
    PV_base: np.ndarray,
    price: np.ndarray,
    export_price: np.ndarray,
    Delta_t: float = 1.0,
    # CapEx (nameplate)
    c_PV_kw: float = 800.0,
    c_bat_E_kWh: float = 150.0,
    c_bat_P_kw: float = 250.0,
    # Efficiencies
    eta_ch: float = 0.95,
    eta_dis: float = 0.95,
    # Optional hard cap on net imports
    P_net_cap: Optional[float] = None,
    # Absolute bounds (tighten for speed)
    P_BAT_ABS_MAX: float = 200_000.0,    # [kW] (was 200 MW)
    E_BAT_ABS_MAX: float = 2_000_000.0,   # [kWh] (was 2000 MWh)
    # Periodic peak charges (e.g., daily)
    peak_tariff_per_MW_period: float = 0.0,     # $/MW per period (scalar) - keep in $/MW for consistency
    peak_period_steps: Optional[int] = None,    # e.g., 24 (hourly→daily), 96 (15-min→daily)
    peak_tariff_profile: Optional[np.ndarray] = None,  # length = #periods (overrides scalar) - keep in $/MW
    # Annualization & scaling
    discount_rate: float = 0.06,
    lifetime_years: float = 15.0,
    replicate_periods: float = 1.0,   # how many times the provided horizon repeats per year
    solver_options: Optional[Dict[str, Any]] = None,
    energy_ratio:float= 0.5,  # P_bat_max / E_bat (1/h)
    **kwargs
) -> Dict[str, Any]:

    # ---- inputs ----
    L = np.asarray(L, float)
    PV_base = np.asarray(PV_base, float)
    price = np.asarray(price, float)
    N = len(L)
    assert PV_base.shape == (N,) and price.shape == (N,)
    PV_base_peak = float(np.max(PV_base))
    if PV_base_peak <= 0 and any(PV_base > 0) and c_PV_kw > 0:
         # Only raise error if there's PV capacity and PV_base is all zero or negative.
         # If c_PV_kw is 0, PV_base_peak doesn't matter for capex.
         if np.max(PV_base) <= 0 and c_PV_kw > 0:
              raise ValueError("PV_base_peak must be > 0 if PV_base has positive values and c_PV_kw > 0")
         else:
              PV_base_peak = 0.0 # Set to 0 if no PV is sized


    # ---- periodization for peak charges ----
    use_period_peaks = (peak_period_steps is not None) and (peak_period_steps > 0) and (
        (peak_tariff_profile is not None and len(peak_tariff_profile) > 0) or (peak_tariff_per_MW_period > 0.0)
    )
    if use_period_peaks:
        P = int(np.ceil(N / peak_period_steps))
        period_of_t = {t: int(t // peak_period_steps) for t in range(N)}
        if peak_tariff_profile is not None:
            tar = np.asarray(peak_tariff_profile, float)
            if len(tar) != P:
                raise ValueError(f"peak_tariff_profile length {len(tar)} != periods {P}")
            peak_tariffs = tar
        else:
            peak_tariffs = np.full(P, float(peak_tariff_per_MW_period), dtype=float)

    # ---- model ----
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, N-1)

    # Vars (sizing)
    m.x_pv      = pyo.Var(bounds=(0, None))             # oversize factor [-]
    m.E_bat     = pyo.Var(bounds=(0, E_BAT_ABS_MAX))    # [kWh]
    m.P_bat_max = pyo.Var(bounds=(0, P_BAT_ABS_MAX))    # [kW]

    # Vars (ops)
    m.P_ch   = pyo.Var(m.T, bounds=(0, P_BAT_ABS_MAX))  # [kW]
    m.P_dis  = pyo.Var(m.T, bounds=(0, P_BAT_ABS_MAX))  # [kW]
    # m.P_curt = pyo.Var(m.T, bounds=(0, None))           # [kW] # Removed
    m.SOC    = pyo.Var(m.T, bounds=(0, E_BAT_ABS_MAX))  # [kWh]
    m.y      = pyo.Var(m.T, domain=pyo.Binary)          # mode: 1=discharge, 0=charge

    # Params
    m.L        = pyo.Param(m.T, initialize={t: float(L[t]) for t in range(N)})
    m.PV_base  = pyo.Param(m.T, initialize={t: float(PV_base[t]) for t in range(N)})
    m.pi       = pyo.Param(m.T, initialize={t: float(price[t]) for t in range(N)}) # $/MWh
    m.pi_export = pyo.Param(m.T, initialize={t: float(export_price[t]) for t in range(N)}) # $/MWh
    m.PV_peak  = pyo.Param(initialize=PV_base_peak) # kW


    # -----------------------------------------------------------------
    # 1) auxiliary variable that will hold the *available* PV surplus
    # -----------------------------------------------------------------
    m.surplus = pyo.Var(m.T, bounds=(0, P_BAT_ABS_MAX))   # kW, non‑negative

    # -----------------------------------------------------------------
    # 2) surplus must be at least the raw excess PV
    # -----------------------------------------------------------------
    def surplus_def_rule(m, t):
        # raw excess = PV output – load + discharge (discharge adds to the surplus)
        return m.surplus[t] >= m.x_pv * m.PV_base[t] - m.L[t]
    m.surplus_def = pyo.Constraint(m.T, rule=surplus_def_rule)

    # -----------------------------------------------------------------
    # 3) charging power cannot exceed the surplus
    # -----------------------------------------------------------------
    def charge_limit_by_surplus_rule(m, t):
        return m.P_ch[t] <= m.surplus[t]
    m.charge_by_surplus = pyo.Constraint(m.T, rule=charge_limit_by_surplus_rule)

    # Net load
    # Removed m.P_curt from the net load calculation
    def net_load_rule(m, t):
        return m.L[t] - (m.x_pv*m.PV_base[t] + m.P_dis[t] - m.P_ch[t]) # kW
    m.P_net = pyo.Expression(m.T, rule=net_load_rule)

    # Power caps & mode (linear Big-M)
    def ch_vs_nameplate(m, t):  return m.P_ch[t]  <= m.P_bat_max
    def dis_vs_nameplate(m, t): return m.P_dis[t] <= m.P_bat_max
    m.ch_nameplate  = pyo.Constraint(m.T, rule=ch_vs_nameplate)
    m.dis_nameplate = pyo.Constraint(m.T, rule=dis_vs_nameplate)

    def ch_mode_rule(m, t):  return m.P_ch[t]  <= P_BAT_ABS_MAX * (1 - m.y[t])
    def dis_mode_rule(m, t): return m.P_dis[t] <= P_BAT_ABS_MAX * m.y[t]
    m.ch_mode  = pyo.Constraint(m.T, rule=ch_mode_rule)
    m.dis_mode = pyo.Constraint(m.T, rule=dis_mode_rule)

    # battery power function of battery size
    def battery_nominal_power(m, t): return m.P_bat_max == m.E_bat * energy_ratio
    m.battery_nom_pow = pyo.Constraint(m.T, rule=battery_nominal_power)


    # Curtailment bound - Removed
    # def curt_bound(m, t): return m.P_curt[t] <= m.x_pv * m.PV_base[t]
    # m.curt_bound = pyo.Constraint(m.T, rule=curt_bound)

    # SOC dynamics

    def soc_dyn(m, t):
        if t < N-1:
            return m.SOC[t+1] == m.SOC[t] + eta_ch*m.P_ch[t]*Delta_t - (1.0/eta_dis)*m.P_dis[t]*Delta_t
        return pyo.Constraint.Skip
    m.soc_dyn = pyo.Constraint(m.T, rule=soc_dyn)
    m.SOC[0].fix(0.0)

    # SOC ≤ E_bat
    def soc_cap(m, t): return m.SOC[t] <= m.E_bat
    m.soc_cap = pyo.Constraint(m.T, rule=soc_cap)

    # Optional hard cap on net import
    if P_net_cap is not None:
        def peak_cap(m, t): return m.P_net[t] <= P_net_cap
        m.peak_cap = pyo.Constraint(m.T, rule=peak_cap)

    # Periodic peak charges
    if use_period_peaks:
        m.P = pyo.RangeSet(0, P-1)
        m.P_peak = pyo.Var(m.P, bounds=(0, None))  # per-period peak import [kW]
        m.period_of_t = pyo.Param(m.T, initialize=period_of_t, within=pyo.Any)
        def epi_rule(m, t):
            p = int(m.period_of_t[t])
            # Convert kW peak to MW peak for the tariff
            return m.P_net[t] / 1000.0 <= m.P_peak[p]
        m.peak_epi = pyo.Constraint(m.T, rule=epi_rule)
        m.peak_tariff = pyo.Param(m.P, initialize={p: float(peak_tariffs[p]) for p in range(P)}) # $/MW per period

    # ---- LCOE (annualized) ----
    # MW_to_kW = 1000.0 # No longer needed as everything is in kW/kWh
    energy_horizon = float(np.sum(L) * Delta_t)                      # kWh over given horizon
    energy_annual  = replicate_periods * energy_horizon

    # Capital Recovery Factor
    r, n = discount_rate, lifetime_years
    crf = (r * (1 + r)**n) / ((1 + r)**n - 1)

    # Variables
    m.P_imp = pyo.Var(m.T, bounds=(0, None))  # imports [kW]
    m.P_exp = pyo.Var(m.T, bounds=(0, None))  # exports [kW]

    # Tie to net
    def net_def(m, t):
        return m.P_net[t] == m.P_imp[t] - m.P_exp[t]
    m.net_def = pyo.Constraint(m.T, rule=net_def)

    # In the objective cost, replace:
    #   opex_energy = sum(m.pi[t] * m.P_net[t] * Delta_t for t in m.T)
    # with:
    price_import  = m.pi          # keep your price vector ($/MWh)
    export_price = m.pi_export           # or a vector/Param with low credit, e.g., 0–5 $/MWh


    def total_annual_cost_rule(m):
        capex = (c_PV_kw * m.x_pv * m.PV_peak # kW * $/kW
                 + c_bat_E_kWh * m.E_bat      # kWh * $/kWh
                 + c_bat_P_kw  * m.P_bat_max) # kW * $/kW
        capex_annual = crf * capex
        # Convert kW to MW for price calculation ($/MWh * MWh)
        opex_energy = sum(price_import[t] * (m.P_imp[t]/1000.0) * Delta_t for t in m.T) \
                - sum(export_price[t] * (m.P_exp[t]/1000.0) * Delta_t for t in m.T)
        # Curtailment cost - Removed
        # curt_cost    = curtail_penalty_per_MWh * sum((m.P_curt[t]/1000.0) * Delta_t for t in m.T)
        peak_cost    = 0.0
        if use_period_peaks:
            peak_cost = sum(m.peak_tariff[p] * (m.P_peak[p]) for p in m.P)
        # replicate operating costs across the year (e.g., 365× for a single-day horizon)
        # Removed curt_cost from opex_annual calculation
        opex_annual = replicate_periods * (opex_energy + peak_cost)
        return capex_annual + opex_annual

    m.total_annual_cost = pyo.Expression(rule=total_annual_cost_rule)
    # Objective is LCOE in $/MWh, so total cost in $ / energy in MWh
    m.obj = pyo.Objective(expr=m.total_annual_cost / (energy_annual / 1000.0), sense=pyo.minimize)

    # ---- solve: Appsi-HiGHS → CBC → GLPK ----
    used_solver, res = None, None

    try:
        from pyomo.contrib.appsi.solvers.highs import Highs as AppsiHighs
        opt = AppsiHighs()
        if solver_options:
            if "time_limit" in solver_options:
                try: opt.config.time_limit = solver_options["time_limit"]
                except: pass
            if "threads" in solver_options:
                try: opt.config.threads = solver_options["threads"]
                except: pass
            if "mip_gap" in solver_options:
                try: opt.config.mip_gap = solver_gap = solver_options["mip_gap"]
                except: pass
        # Remove options that caused fallback
        # opt.highs_options['time_limit'] = 200
        # opt.options['time_limit'] = 200
        res = opt.solve(m)
        used_solver = "appsi_highs"
    except Exception as e:
        print("[appsi_highs] fallback:", e)

    if used_solver is None:
        import shutil
        try:
            cbc_path = shutil.which("cbc") or "/usr/bin/cbc"
            solver = pyo.SolverFactory("cbc", executable=cbc_path)
            if solver_options:
                if "ratio" in solver_options:   solver.options["ratio"]  = solver_options["ratio"]
                if "seconds" in solver_options: solver.options["seconds"] = solver_options["seconds"]
            res = solver.solve(m)
            used_solver = "cbc"
        except Exception as e:
            print("[cbc] fallback:", e)

    if used_solver is None:
        import shutil
        try:
            glpsol_path = shutil.which("glpsol") or "/usr/bin/glpsol"
            solver = pyo.SolverFactory("glpk", executable=glpsol_path)
            res = solver.solve(m)
            used_solver = "glpk"
        except Exception as e:
            print("[glpk] fallback:", e)

    if used_solver is None:
        raise RuntimeError("No working MILP solver.")

    # ---- results ----
    v = pyo.value
    def _termination_str(res_obj):
        tc = getattr(res_obj, "termination_condition", None)
        if tc is not None: return str(tc)
        solver_obj = getattr(res_obj, "solver", None)
        if solver_obj is not None and hasattr(solver_obj, "termination_condition"):
            return str(solver_obj.termination_condition)
        return "unknown"

    out = {
        "solver": used_solver,
        "termination": _termination_str(res),
        "LCOE_$/MWh": v(m.obj), # Keep LCOE in $/MWh as it is a standard metric
        "total_annual_cost_$": v(m.total_annual_cost),
        "x_pv": v(m.x_pv),
        "E_bat_kWh": v(m.E_bat), # kWh
        "P_bat_max_kW": v(m.P_bat_max), # kW
        "PV_base_peak_kW": PV_base_peak, # kW
        "series": {
            "P_net_kW":   np.array([v(m.P_net[t])  for t in m.T]), # kW
            "P_ch_kW":    np.array([v(m.P_ch[t])   for t in m.T]), # kW
            "P_dis_kW":   np.array([v(m.P_dis[t])  for t in m.T]), # kW
            "SOC_kWh":    np.array([v(m.SOC[t])    for t in m.T]), # kWh
            # "curtail_kW": np.array([v(m.P_curt[t]) for t in m.T]), # kW # Removed
            "y_mode":     np.array([v(m.y[t])      for t in m.T]),
        },
        "scaling": {
            "CRF": crf,
            "replicate_periods": replicate_periods,
            "energy_annual_kWh": energy_annual, # kWh
        }
    }
    if use_period_peaks:
        out["period_peaks_MW"] = np.array([v(m.P_peak[p]) for p in m.P]) # Still in MW due to tariff
        out["peak_tariff_per_period"] = peak_tariffs # Still in $/MW
        out["peak_period_steps"] = peak_period_steps
    return out
