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
    plt.savefig("battery_sizing_cfa/figs/daily_max_quantiles_distribution_sizing_{}_{}.pdf".format(specs['sizing_method'], billing_peak_period_str))

    return day_max_quantiles


def analyze_results_mpc_vs_rbc(res_path, specs):
    import pickle
    with open(res_path, 'rb') as f:
        results = pickle.load(f)

    extract_day_max_quantiles_over_meters(results, specs=specs)


