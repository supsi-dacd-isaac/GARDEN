import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_results(x_pv, E_bat_kWh, P_battery, SOC_kWh, PV_base, L, price):
    N = len(L)
    scaled_pv = x_pv * PV_base
    fig = go.Figure()
    time_steps = np.arange(N)
    fig.add_trace(go.Scatter(x=time_steps, y=L, mode='lines', name='Load (kW)'))
    fig.add_trace(go.Scatter(x=time_steps, y=L-scaled_pv, mode='lines', name='Net Load (kW)'))
    fig.add_trace(go.Scatter(x=time_steps, y=L-scaled_pv+P_battery, mode='lines', name='Optimized Net Load (kW)'))
    fig.add_trace(go.Scatter(x=time_steps, y=P_battery, mode='lines', name='Battery (kW)'))
    fig.add_trace(go.Scatter(x=time_steps, y=price, mode='lines', name='Price ($/MWh)'))
    fig.add_trace(go.Scatter(x=time_steps, y=SOC_kWh/E_bat_kWh*100 , mode='lines', name='State of Charge (kWh)'))

    # Update layout
    fig.update_layout(
        title='Load, PV, and Battery Operations',
        xaxis_title='Time Step',
        yaxis_title='Power (kW)',
        hovermode='x unified',
    )
    # set white background
    fig.update_layout(plot_bgcolor='white')
    # set white template
    fig.update_layout(template='plotly_white')

    # set fig dimensions
    fig.update_layout(width=1000, height=400)
    return fig

def plot_rbc_vs_mpc_diagnostics(x_test, results, series, billing_peak_period_str, sizing_method):
    p_grid_rbc = results['profiles']['rbc']
    p_grid_mpc =  results['profiles']['mpc']
    p_grid_mpc_opt = results['profiles']['mpc_opt']
    target_name = results['target_name']

    fig, ax = plt.subplots(6, 1, figsize=(12, 8), layout='constrained')
    w_len = len(x_test)//6
    # retrieve standard color map
    colors = plt.get_cmap('tab10')
    for i, a in enumerate(ax.ravel()):
        w = np.arange(w_len) + i * w_len
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.plot(x_test.index.values[w], x_test.iloc[w][target_name].values, label='Load', alpha=1, linewidth=0.5, color=colors(0))
        for j, k in enumerate(results['profiles'].keys()):
            a.plot(x_test.index.values[w], results['profiles'][k][w], label=k, alpha=1, linewidth=0.5, color=colors(j+1))
        a.set_ylabel('Power')
        a.set_title('Grid Power Profiles on Test Set')
        a.legend(fontsize='small', loc='upper right', ncol=2)
        a.set_xlabel('Time')

    plt.savefig("battery_sizing_cfa/figs/rbc_vs_mpc_profiles_{}_billed_{}.pdf".format( series, billing_peak_period_str))

    # daily maxima distribution comparison using seaborn

    df_dm = pd.DataFrame(results['daily_maxima'])
    df_dm_melted = df_dm.melt(var_name='Method', value_name='Daily Maxima')
    plt.figure(figsize=(8, 6), layout='constrained')
    # boxenplot sorted by median
    sns.boxenplot(data=df_dm_melted, x='Method', y='Daily Maxima', order=df_dm.median().sort_values().index,
                  fill=False, linewidth=.5, linecolor=".7",
                    line_kws=dict(linewidth=1.5, color="#cde"),
                    flier_kws=dict(facecolor=".7", linewidth=.5))

    # despine the plot
    sns.despine()
    plt.title('Daily Maxima Distribution Comparison')
    plt.savefig("battery_sizing_cfa/figs/rbc_vs_mpc_daily_maxima_{}_{}_billed_{}.pdf".format(series, sizing_method, billing_peak_period_str))
    plt.close('all')



def extract_day_max_quantiles_over_meters(results, normalize_quantiles=True, normalize_with_opt_mpc=True, specs=None):

    # obtain a multicolumn dataframe: index = quantiles, columns = meter ids, column level 1 = method (rbc, mpc, mpc_opt)
    quantile_levels = [5, 25, 50, 75, 95, 99]

    # collect quantiles for each (meter_id, method)
    cols = {}
    for meter_id, record in results.items():
        dm = record.get('daily_maxima', {})  # dict: method -> daily_maxima_array
        for method, daily_vals in dm.items():
            cols[(meter_id, method)] = np.percentile(daily_vals, quantile_levels)
            if normalize_quantiles:
                cols[(meter_id, method)] /= record.get('mean_abs_val', 1.0)

    # DataFrame with MultiIndex columns: (meter_id, method)
    df_multi = pd.DataFrame(cols, index=quantile_levels)
    df_multi.columns = pd.MultiIndex.from_tuples(df_multi.columns, names=['meter_id', 'method'])

    # Optional: dict mapping method -> DataFrame (columns = meter_ids)
    if normalize_with_opt_mpc:
        day_max_quantiles = {
            method: df_multi.xs(method, axis=1, level='method').copy() / df_multi.xs('mpc_opt', axis=1, level='method').copy()
            for method in df_multi.columns.get_level_values('method').unique() if method != 'mpc_opt'
        }
    else:
        day_max_quantiles = {
            method: df_multi.xs(method, axis=1, level='method').copy()
            for method in df_multi.columns.get_level_values('method').unique()
        }

    day_max_quantiles = pd.concat(day_max_quantiles, axis=1).melt(ignore_index=False).reset_index().rename(columns={'index': 'quantile', None:'controller'})
    # day_max_quantiles cols are ['quantile', 'controller','meter_id','value'] plot the distributions of value for each (quantile, controller) using seaborn boxenplot
    plt.figure(figsize=(10, 6), layout='constrained')
    sns.boxenplot(data=day_max_quantiles, x='quantile', y='value', hue='controller')
    plt.title('Distribution of Normalized Daily Maxima Quantiles Across Meters')
    plt.xlabel('Quantile Level')
    plt.ylabel('Normalized Daily Maxima Value')
    plt.legend(title='Controller Method')
    # despine
    sns.despine()
    billing_peak_period_str = specs['billing_peak_period_str']
    plt.savefig("battery_sizing_cfa/figs/daily_max_quantiles_distribution_sizing_{}_{}.png".format(specs['sizing_method'], billing_peak_period_str))

    return day_max_quantiles

def analyze_sizing(results, specs):
    sizings = pd.Series({k: r['E_bat_kWh'] for k, r in results.items()}, name='{}_{}'.format(specs['sizing_method'], specs['billing_peak_period_str']))
    return sizings

def analyze_lcoes(results, specs):
    lcoes = pd.DataFrame({k: pd.Series(r['lcoe']) for k, r in results.items()})
    lcoes_sizing = pd.Series({k: r['lcoe_sizing'] for k, r in results.items()}, name='lcoe_sizing')
    lcoes = pd.concat([lcoes.T, lcoes_sizing], axis=1).T

    # Prepare data once
    plot_data = lcoes.T.melt(ignore_index=False).reset_index().rename(
        columns={'index': 'series', 'value': 'lcoe_value', 'variable': 'controller'}
    )

    # Create mapping: series → lcoe_sizing value
    sizing_map = plot_data[plot_data['controller'] == 'lcoe_sizing'].set_index('series')['lcoe_value']

    # PLOT SETUP
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')

    # PLOT 1: Full original distribution
    sns.boxenplot(data=plot_data, x='controller', y='lcoe_value', hue='controller', ax=ax1)
    ax1.set_title('LCOE Distributions', fontweight='bold')
    ax1.set_ylabel('LCOE ($/MWh)')
    sns.despine(ax=ax1)

    # PLOT 2: Normalized data
    # Filter out lcoe_sizing and divide by corresponding sizing values
    norm_data = plot_data[plot_data['controller'] != 'lcoe_sizing'].copy()
    norm_data['lcoe_value'] = norm_data.apply(
        lambda row: row['lcoe_value'] / sizing_map[row['series']], axis=1
    )

    sns.boxenplot(data=norm_data, x='controller', y='lcoe_value', hue='controller', ax=ax2)
    ax2.set_title('Normalized (relative to LCOE Sizing)', fontweight='bold')
    ax2.set_ylabel('LCOE Ratio')
    ax2.set_ylim(0., 2)

    sns.despine(ax=ax2)

    # Shared formatting
    for ax in [ax1, ax2]:
        ax.set_xlabel('Sizing Method')

    plt.savefig("battery_sizing_cfa/figs/lcoe_sizing_comparison_{}_{}.pdf".format(specs['sizing_method'], specs['billing_peak_period_str']))
    plt.close('all')

def analyze_results_mpc_vs_rbc(res_path, specs):
    import pickle
    with open(res_path, 'rb') as f:
        results = pickle.load(f)

    day_max_quantiles_df =  extract_day_max_quantiles_over_meters(results, specs=specs, normalize_quantiles=True, normalize_with_opt_mpc=True)
    analyze_lcoes(results, specs)
    sizings = analyze_sizing(results, specs)

    return day_max_quantiles_df, sizings

if __name__ == "__main__":
    hours_prescient_sizing = 24 * 30
    specs = {'eta_ch': 0.95,
             'eta_dis': 0.95,
             'soc_min': 0,
             'soc_max': 1,
             'soc_start': 0.2,
             'peak_tariff_per_MW_period': 500,
             'energy_ratio': 1.0,
             'peak_period_steps': 24,
             'c_PV_kw': 200,
             'c_bat_E_kWh': 120,
             'c_bat_P_kw': 50,
             'hours_prescient_sizing': hours_prescient_sizing,
             'replicate_periods': int(np.ceil(24 * 365 / (hours_prescient_sizing))),
             'Delta_t': 1.0,
             'discount_rate': 0.06,
             'lifetime_years': 15.0,
             'billing_peak_period_str': 'daily',
             'sizing_method': 'prescient'
             }
    from os.path import join
    res_path = "battery_sizing_cfa/results/"
    mp_tuples = (('prescient', 'monthly'), ('rbc_peak_shaving', 'monthly'), ('prescient', 'daily'))
    dmq_dfs = {}
    sizing = {}
    for method, period in mp_tuples:
        file_path = 'rbc_vs_mpc_results_{}_peaks_billed_{}.pk'.format(method, period)
        specs.update({'sizing_method': method, 'billing_peak_period_str': period})
        key = '{}_{}'.format(method, period)
        dmq_dfs[key], sizing[key]  = analyze_results_mpc_vs_rbc(join(res_path, file_path), specs=specs)

    dmq_dfs_comb = pd.concat(dmq_dfs.values(), keys=dmq_dfs.keys(), names=['method_period'], axis=0)
    dmq_dfs_comb.drop('prescient_daily', inplace=True)
    dmq_dfs_comb = dmq_dfs_comb.rename(index={'prescient_monthly': 'A', 'rbc_peak_shaving_monthly': 'B'})
    dmq_dfs_comb = dmq_dfs_comb.reset_index(drop=False)

    dmq_dfs_comb['quantile_third'] = dmq_dfs_comb.apply(
        lambda row: f"{row['controller']}-{row['method_period']}", axis=1
    )

    plt.figure(figsize=(14, 6))

    # Define same color with different alpha values for each controller
    # Adjust RGB values as needed (here: steel blue)
    controllers = dmq_dfs_comb['quantile_third'].unique()
    colors = plt.get_cmap('tab10', len(controllers))
    palette = {ctrl: np.clip(np.array(colors(i%dmq_dfs_comb['controller'].nunique())[:-1]) * (1 + 0.5*(i//dmq_dfs_comb['controller'].nunique())),0, 1) for i, ctrl in enumerate(controllers)}

    # Create the plot
    sns.boxenplot(
        data=dmq_dfs_comb,
        x='quantile',
        y='value',
        hue='quantile_third',
        dodge=0.6,  # Controls separation: 0=overlapping, 1=full separation
        palette=palette,
        linewidth=1.2,
        width=0.8   # Controls box width within each group
    )

    # Formatting
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Quantile')
    plt.ylabel('Normalized peaks')
    plt.title('Normalized Quantile distributions across meters')
    plt.legend(title='Controller', ncol=2)
    plt.tight_layout()
    plt.semilogy()

    plt.savefig("battery_sizing_cfa/figs/daily_max_quantiles_distribution_all_methods.pdf")


    sizing_df = pd.DataFrame(sizing).iloc[:, :-1]
    # boxenplot of sizing results
    plt.figure(figsize=(8, 6), layout='constrained')
    sns.boxenplot(data=sizing_df, palette='Set2', linewidth=1.2)
    plt.ylabel('Battery Energy Capacity (kWh)')
    plt.title('Battery Sizing Comparison Across Methods')
    sns.despine()
    plt.savefig("battery_sizing_cfa/figs/battery_sizing_comparison_all_methods.pdf")


