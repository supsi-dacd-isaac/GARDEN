import numpy as np
import plotly.graph_objects as go


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
    # set fig dimensions
    fig.update_layout(width=1000, height=400)
    return fig