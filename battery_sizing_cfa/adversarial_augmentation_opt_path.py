from battery_sizing_cfa.optimizers.peak_shaver import optimize_peak_shaving
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from battery_sizing_cfa.optimizers.rbc_sizing import rbc_peak_shaving
from battery_sizing_cfa.optimizers.parametric_rbc import rbc_thresholds
from battery_sizing_cfa.cost_functions.peak_shaving import daily_maxima, day_max_cost_from_results
from battery_sizing_cfa.cost_functions.utils import build_block_basis, cvar_from_daily_losses, frac_rolling_quantile
from matplotlib import cm
from matplotlib.colors import Normalize
from time import time

zlims = (58, 65)
data = pd.read_pickle("battery_sizing_cfa/datasets/portugal/portugal.pk")

specs = {'c_bat_E_kwh': 20.0,
         'energy_ratio': 1.0,
         'eta_ch': 0.95,
         'eta_dis': 0.95,
         'soc_min':0,
         'soc_max':1,
         'soc_start':0.2,
         'adversarial_budget': 100,
         'adversarial_perturbation': True}

bounds = [
    (0, 0.9),  # lower_q
    (0.1, 1),  # higher_q
    (5, 24*7), # n_hours
]



series = 8
n_tr = 365*24
n_te = 365*24
adv_max = 0.99
L_tr = data.iloc[:n_tr, series].values/np.std(data.iloc[:n_tr, series])  # Load and PV generation data
L_te = data.iloc[n_tr:n_tr+n_te, series].values/np.std(data.iloc[:n_tr, series]) # Load and PV generation data

fig, ax = plt.subplots(2, 1, figsize=(15, 4), layout='constrained')
ax[0].plot(L_tr, label='p train', linewidth=1)
ax[1].plot(L_te, label='p test', linewidth=1)
plt.savefig("battery_sizing_cfa/figs/tr_te_profiles_{}.png".format(series), dpi=100, bbox_inches="tight")
plt.close('all')

hour_index_tr = data.index[:n_tr].hour.values
hour_index_te = data.index[n_tr:n_tr+n_te].hour.values
price_tr = np.array([40 if (t%24) in range(18,24) else 20 for t in range(n_tr)])  # $/MWh
price_te = np.array([40 if (t%24) in range(18,24) else 20 for t in range(n_te)])  # $/MWh


trials = 10
score_tr = np.full((trials), np.nan)
score_te = np.full((trials), np.nan)
opt_pars = np.full((trials, 3), np.nan)
score_te_daily = []

rbc_peak_shaving_wrap = lambda x, L_tr, price, specs: rbc_peak_shaving({'lower_q':x[0], 'higher_q':x[1], 'n_hours':x[2]}, L_tr, price, specs, h=hour_index_tr)

for num, ad in enumerate(np.linspace(0, adv_max, trials)):
    specs['alpha_cvar'] = ad

    t_tot = 0
    # optimize rbc_peak_shaving via differential evolution
    from scipy.optimize import differential_evolution, shgo
    t0 = time()

    result = differential_evolution(
        rbc_peak_shaving_wrap,
        #x0 = [0.3, 0.7, 24],
        bounds=bounds,
        args=(L_tr, price_tr, specs),
        init='random',
        strategy='best1bin',
        maxiter=100,
        popsize=50,
        tol=0.01,
        polish=False,
        integrality=(False, False, False)
    )
    """
    result = shgo(
        rbc_peak_shaving_wrap, bounds=bounds, n=100, iters=3, args=(L_tr, price_tr, specs))
    """
    score_tr[num] = result.fun
    opt_pars[num] = result.x
    # performance on test set
    best_q_low, best_q_high, best_n = result.x
    #lower_threshold = pd.Series(np.concat([L_tr, L_te])).rolling(window=int(best_n), min_periods=1).quantile(best_q_low).to_numpy()[-len(L_te):]
    #upper_threshold = pd.Series(np.concat([L_tr, L_te])).rolling(window=int(best_n), min_periods=1).quantile(best_q_high).to_numpy()[-len(L_te):]
    L_all = np.concatenate([L_tr, L_te])
    lt_all = frac_rolling_quantile(L_all, W_star=best_n, q=best_q_low)
    lower_threshold = lt_all[-len(L_te):]
    ut_all = frac_rolling_quantile(L_all, W_star=best_n, q=best_q_high)
    upper_threshold = ut_all[-len(L_te):]

    print("Best q_low = {:0.2f}, q_high = {:0.2f}, n_hours = {:0.2f}".format(best_q_low, best_q_high, best_n))

    p_battery = specs.get('c_bat_E_kwh', 1.0) * specs.get('energy_ratio', 1.0)

    soc, p_batt, p_grid = rbc_thresholds(
        L_te,                     # net consumption array
        L_te * 0,                 # placeholder PV flag (kept for signature)
        capacity_kwh=specs.get('c_bat_E_kwh', 1.0),
        soc_start=specs.get('soc_start', 0.5),
        soc_min=specs.get('soc_min', 0.1),
        soc_max=specs.get('soc_max', 0.99),
        p_charge_max=p_battery,
        p_discharge_max=p_battery,
        eta_ch=specs.get('eta_ch', 0.99),
        eta_dis=specs.get('eta_dis', 0.99),
        dt_hours=1.0,
        noise_level=0,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold
    )

    daily_max_base = daily_maxima(p_grid, hour_index_te)
    score_te[num] = cvar_from_daily_losses(daily_max_base, specs['alpha_cvar'])
    score_te_daily.append(daily_maxima(p_grid, hour_index_te))
    fig, ax = plt.subplots(1, 1, figsize=(15, 4))
    ax.plot(L_te, label='p')
    ax.plot(L_te+p_batt, label='p controlled')
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.title('adversarial budget: {:03f}, of:{:0.2e}'.format(ad, np.mean(p_grid**2)))
    plt.legend(loc='upper right')
    out = f"battery_sizing_cfa/figs/adv_sols_{series}_{ad:03f}.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close('all')


    plt.hist(daily_max_base, bins=30, alpha=0.5, label='base')
    if num==0:
        xlims_dm = plt.xlim()
        ylims_dm = plt.ylim()

    plt.xlim(xlims_dm[0]*0.95, xlims_dm[1]*1.1)
    plt.ylim(ylims_dm[0], ylims_dm[1]*1.1)
    p95_base = np.percentile(daily_max_base, 95)
    p99_base = np.percentile(daily_max_base, 99)
    plt.axvline(p95_base, color='blue', linestyle='--', label='95% base')
    plt.axvline(p99_base, color='blue', linestyle=':', label='99% base')
    plt.legend()
    plt.title('Daily maxima distribution under adversarial perturbation')
    out = f"battery_sizing_cfa/figs/adv_daily_maxima_{series}_{ad:03f}.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close('all')

# plot 3d path of best params
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter(opt_pars[:,0], opt_pars[:,1], opt_pars[:,2], c=score_te, cmap='viridis', s=100)
for i in range(trials-1):
    ax.plot(opt_pars[i:i+2,0], opt_pars[i:i+2,1], opt_pars[i:i+2,2], color='gray', alpha=0.5)
ax.set_xlabel('Q low')
ax.set_ylabel('Q high')
ax.set_zlabel('n hours')
plt.title('Adversarial Augmentation Optimization Path')
cbar = plt.colorbar(scat, ax=ax, pad=0.1)
cbar.set_label('Peak Shaving Cost')
plt.show()

# plot score_te_daily quantiles as a function of adversarial budget
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
adversarial_budgets = np.linspace(0, adv_max, trials)
quantiles = [5, 25, 50, 75, 95, 99]
for q in quantiles:
    q_values = [np.percentile(score_te_daily[i], q) for i in range(trials)]
    ax.plot(adversarial_budgets, q_values, label=f'{q}th Percentile')
ax.set_xlabel('Adversarial Budget')
ax.set_ylabel('Daily Maximum Cost')
ax.set_title('Daily Maximum Cost Quantiles vs Adversarial Budget')
ax.legend(loc='lower left')
plt.show()