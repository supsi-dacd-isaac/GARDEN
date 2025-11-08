import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from battery_sizing_cfa.optimizers.rbc_sizing import rbc_peak_shaving
from battery_sizing_cfa.optimizers.parametric_rbc import rbc_thresholds
from battery_sizing_cfa.cost_functions.peak_shaving import daily_maxima, day_max_cost_from_results

zlims = (58, 65)
data = pd.read_pickle("battery_sizing_cfa/datasets/portugal/portugal.pk")

specs = {'c_bat_E_kWh': 20.0,
         'energy_ratio': 1.0,
         'eta_ch': 0.95,
         'eta_dis': 0.95,
         'soc_min':0,
         'soc_max':1,
         'soc_start':0.2,
         'adversarial_budget': 100,
         'adversarial_perturbation': True}



from matplotlib import cm
from matplotlib.colors import Normalize
series = 1
n_tr = 200*24
n_te = 165*24
L_tr = data.iloc[:n_tr, series].values/np.std(data.iloc[:n_tr, 0])  # Load and PV generation data
L_te = data.iloc[n_tr:n_tr+n_te, series].values/np.std(data.iloc[:n_tr, 0])  # Load and PV generation data
hour_index_tr = data.index[:n_tr].hour.values
hour_index_te = data.index[n_tr:n_tr+n_te].hour.values


price_tr = np.array([40 if (t%24) in range(18,24) else 20 for t in range(n_tr)])  # $/MWh
price_te = np.array([40 if (t%24) in range(18,24) else 20 for t in range(n_te)])  # $/MWh

q_low_range = np.linspace(0.1, 0.7, 15) # Adjust the range as needed
q_high_range = np.linspace(0.5, 1, 15)   # Adjust the range as needed
n_hours_range = np.linspace(5, 24*7, 10).astype(int)

# Create a grid of parameter values
Q_LOW, Q_HIGH, N_HOURS = np.meshgrid(q_low_range, q_high_range, n_hours_range)
n_q_low, n_q_high, n_hours = Q_LOW.shape
PS_cube = np.full((n_q_low, n_q_high, n_hours), np.nan)
for num, ad in enumerate(np.linspace(0, 200, 40)):
    specs['adversarial_budget'] = int(ad)  # total budget
    specs["adversarial_magnitude"] = np.mean(L_tr)*0.5
    PS_cube = np.full((n_q_low, n_q_high, n_hours), np.nan)
    from time import time
    t_tot = 0
    for i in range(n_q_low):
        for j in range(n_q_high):
            for k in range(n_hours):
                t0 = time()
                params = {
                    'lower_q': float(Q_LOW[i, j, k]),
                    'higher_q': float(Q_HIGH[i, j, k]),
                    'n_hours': int(N_HOURS[i, j, k])
                }
                PS_cube[i, j, k], L_adv = rbc_peak_shaving(params, L_tr, price_tr, specs, return_adv=True, h=hour_index_tr)
                t_tot  += time()-t0
    plt.plot(L_tr, linewidth=0.5, alpha=0.5)
    plt.plot(L_adv, linewidth=0.5, alpha=0.5)
    plt.savefig(f"battery_sizing_cfa/figs/adv_perturbs_{series}_{ad:03f}.png")

    plt.close('all')
    print('average time : {:0.2e}'.format(t_tot/(n_q_low*n_q_high*n_hours)))
    # collapse the N_HOURS dimension by taking the minimum over k
    PS_min = np.nanmin(PS_cube, axis=2)

    # find argmin indices along the N_HOURS axis and map to actual N_HOURS values
    best_k = np.nanargmin(PS_cube, axis=2)               # indices into k dimension
    best_n_hours = np.empty((n_q_low, n_q_high), dtype=int)
    for i in range(n_q_low):
        for j in range(n_q_high):
            best_n_hours[i, j] = int(N_HOURS[i, j, best_k[i, j]])

    # Prepare 2D grids for plotting (Q_LOW and Q_HIGH are constant along k)
    Q_LOW_2D = Q_LOW[:, :, 0]
    Q_HIGH_2D = Q_HIGH[:, :, 0]

    # ensure arrays
    X = np.asarray(Q_LOW_2D)
    Y = np.asarray(Q_HIGH_2D)
    Z = np.asarray(PS_min)
    C = np.asarray(best_n_hours)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    cmap = cm.get_cmap("viridis")
    norm = Normalize(vmin=np.nanmin(C), vmax=np.nanmax(C))
    facecolors = cmap(norm(C))

    # plot surface with mapped facecolors
    ax.plot_surface(X, Y, Z, facecolors=facecolors, linewidth=0, antialiased=True, alpha=.5)

    # colorbar (use a mappable so it shows the same scale)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(C)
    fig.colorbar(mappable, ax=ax, shrink=0.6, label="best n_hours")


    ax.set_xlabel("Q low [-]")
    ax.set_ylabel("Q high [-]")
    ax.set_zlabel("Peak shaving")
    #ax.set_zlim(zlims[0], zlims[1])
    plt.title('adversarial budget: {:03f}'.format(ad))
    # save PNG
    if num == 0:
        zlims = (np.nanmin(Z), np.nanmax(Z))
    ax.set_zlim(zlims[0]*0.95, zlims[1]*1.15)
    # add a star where the minimum is
    idx_min = np.unravel_index(np.nanargmin(Z), Z.shape)
    ax.scatter(X[idx_min], Y[idx_min], Z[idx_min], color='red', s=100, marker='*', label='Minimum')
    plt.legend()

    out = f"battery_sizing_cfa/figs/adv_landscape_{series}_{ad:03f}.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)

    # line plot on the test set
    idx = np.unravel_index(np.nanargmin(PS_min), PS_min.shape)
    best_q_low = float(Q_LOW_2D[idx])
    best_q_high = float(Q_HIGH_2D[idx])
    best_n = int(best_n_hours[idx])
    # build rolling thresholds (use integer window)
    lower_threshold = pd.Series(np.concat([L_tr, L_te])).rolling(window=int(best_n), min_periods=1).quantile(best_q_low).to_numpy()[-len(L_te):]
    upper_threshold = pd.Series(np.concat([L_tr, L_te])).rolling(window=int(best_n), min_periods=1).quantile(best_q_high).to_numpy()[-len(L_te):]

    print("Best q_low = {:0.2f}, q_high = {:0.2f}, n_hours = {:f}".format(best_q_low, best_q_high, best_n))

    p_battery = specs.get('E_bat_kWh', 1.0) * specs.get('energy_ratio', 1.0)

    soc, p_batt, p_grid = rbc_thresholds(
        L_te,                     # net consumption array
        L_te * 0,                 # placeholder PV flag (kept for signature)
        capacity_kwh=specs.get('E_bat_kWh', 1.0),
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
    fig, ax = plt.subplots(1, 1, figsize=(15, 4))
    ax.plot(L_te, label='p')
    ax.plot(L_te+p_batt, label='p controlled')
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.title('adversarial budget: {:03f}, of:{:0.2e}'.format(ad, day_max_cost_from_results(p_grid, hour_index_te)))
    plt.legend(loc='upper right')
    out = f"battery_sizing_cfa/figs/adv_sols_{series}_{ad:03f}.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close('all')

    # plot distribution of daily maxima with 95% percentile vertical line

    daily_max_base = daily_maxima(p_grid, hour_index_te)
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