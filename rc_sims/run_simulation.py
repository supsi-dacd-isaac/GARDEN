import time
import os
import json
import pandas as pd
import numpy as np
import pickle as pk
import logging
from rc_sims.simulator.simulator_interface import SimulatorInterface


cfg = json.load(open('rc_sims/conf_generator/outputs/paired_baseline_sims_ail/sim/paired_garden_m800_hp59_dhw41_baseline_2030-01-01_2029-12-31_0000-0799_2024-06-10_12-34.json'))
sim_interface = SimulatorInterface(sim_name='{}_{}'.format(cfg['id'], sim_name), sim_base_conf='private_confs/garden_long_merged.json',
                                           sim_interface_conf=cfg['sim_interface'], df_client=df_client, dt=dt)




def simulate(cfg, cfg_metadata, times, sim_name, figs_dir, pks_dir, meter_dicts, flexi_estimator, df_dso, df_meteo, df_client):

    dt = cfg['dt']
    baseline_in_p_hat = cfg['baseline_in_p_hat']
    show_plots = cfg['show_plots']
    save_figs = cfg['save_figs']
    res_dir = cfg['res_dir']
    res_filename = cfg['res_filename']
    scenario_path = cfg['scenario_path']
    metamodel = cfg['metamodel']
    optimization_type = cfg['optimization_type']
    loss = cfg['objective'] if 'objective' in cfg else 'tou_peak'
    solver_pars = cfg['solver_pars'] if 'solver_pars' in cfg else None
    print('{} started'.format(sim_name))

    if exists(join(pks_dir, 'group_responses_all_{}.zip'.format(times[-1]))):
        print('{} already processed'.format(sim_name))
        return
    # load oracle results
    if not metamodel:

    res_path = join(res_dir, res_filename)
    with open(res_path, 'rb') as f:
        results = pk.load(f)

    if 'original_cv_results' in results.keys():
        oracles = results['oracles']
    oracle = oracles['grid_LGBEnergyAware']
    #oracle = oracles['grid_LGBMHybrid']

    controller = BruteForceController(scenario_path, optimization_vars=None, oracle=oracle,
                                      p_tot_forecaster=IdentityForecaster(), loss=LOSSES_MAP[optimization_type][loss],
                                      saved_model_dir=join('oracle/saved_models/', res_filename.split('.')[0]),
                                      treelite=cfg['treelite'] if 'treelite' in cfg else False, hummingbird=cfg['hummingbird'] if 'hummingbird' in cfg else False,
                                      force_recompile=cfg['force_recompile'] if 'force_recompile' in cfg else False)
    oracle['model'].formatter.logger.setLevel(logging.WARNING)

    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(pks_dir, exist_ok=True)
    _, time_specs = get_vars_specs(oracle)
    max_hist_meteo = time_specs.loc[['mean_GHI', 'mean_T'], 'start_time'].min()
    max_future_meteo = time_specs.loc[['mean_GHI', 'mean_T'], 'end_time'].max()

    last_day_history = None
    df_dso_measured = df_dso['power_lag_-001']
    df_controlled = pd.DataFrame()
    df_controlled_oracle = pd.DataFrame()
    df_force_off = pd.DataFrame()
    group_responses_all = pd.DataFrame(columns=pd.MultiIndex.from_product([meter_dicts.keys(), ['tot', 'diff']]))

    last_peaks = {m: 0 for m in np.arange(1, 13)}

    min_h = {}
    max_h_off = {}
    for g, meter_dict in meter_dicts.items():
        min_h[g] = flexi_estimator.estimate_minimum_hours(meter_dict['meters'], meteo_data=df_meteo.loc[
            (df_meteo.index < times[-1] + pd.Timedelta('1D') + pd.Timedelta('{}s'.format(dt))) & (
                        df_meteo.index >= times[0] + pd.Timedelta('{}s'.format(dt)))].resample('1D').mean().rename(
            columns={'mean_GHI': 'GHI', 'mean_T': 'T'}))
        max_h_off[g] = np.floor((24 - min_h[g].quantile(0.99, axis=1)) * 4) / 4

    for t in times:
        t0 = time.time()

        group_responses_all_path = join(pks_dir, 'group_responses_all_{}.zip'.format(t))
        if exists(group_responses_all_path):
            print('{} already processed for time {}'.format(sim_name, t))
            next_day_path = join(pks_dir, 'group_responses_all_{}.zip'.format(t+pd.Timedelta('1D')))
            if not exists(next_day_path):
                group_responses_all = pd.read_pickle(group_responses_all_path)
                last_day_history = group_responses_all.loc[:, (slice(None), 'tot')].droplevel(1, axis=1)
                df_controlled = df_controlled.combine_first(last_day_history)
                df_force_off_path = join(pks_dir, 'df_force_off_{}.zip'.format(t))
                df_force_off = pd.read_pickle(df_force_off_path)
            continue
        for g, meter_dict in meter_dicts.items():
            # min_h = flexi_estimator.estimate_minimum_hours(meter_dict['meters'], meteo_data=df_meteo.loc[
            #     (df_meteo.index < t + pd.Timedelta('1D') + pd.Timedelta('{}s'.format(dt))) & (df_meteo.index >= t + pd.Timedelta('{}s'.format(dt)))].resample('1D').mean().rename(
            #     columns={'mean_GHI': 'GHI', 'mean_T': 'T'}))
            # max_h_off = np.floor((24 - min_h.quantile(0.99, axis=1))*4).values[0]/4
            meter_dicts[g]['max_h_off'] = np.minimum(max_h_off[g].loc[t + pd.Timedelta('{}s'.format(dt))], meter_dicts[g]['max_h_off_lim'])

        df_meteo_t = df_meteo.loc[(df_meteo.index < t + max_future_meteo) & (df_meteo.index > t+max_hist_meteo)]
        df_power_t = pd.DataFrame(0, index=pd.date_range(
            t + time_specs.loc['total consumption', 'start_time'],
            t + time_specs.loc['total consumption', 'end_time'] - pd.Timedelta('{}s'.format(dt)), freq='{}s'.format(dt)), columns=['total consumption'])

        if last_day_history is None:
            df_force_off_t = pd.DataFrame(0, index=pd.date_range(
            t + time_specs.loc['force_off_long', 'start_time'],
            t + time_specs.loc['force_off_long', 'end_time'] - pd.Timedelta('{}s'.format(dt)), freq='{}s'.format(dt)), columns=['force_off_long'])

            df_t = df_meteo_t.combine_first(df_power_t)
            df_t = df_t.combine_first(df_force_off_t)
            df_t = pd.concat({group_name: df_t for group_name in meter_dicts.keys()}, axis=1)
        else:
            base_power = pd.DataFrame({g: df_power_t['total consumption'] for g in meter_dicts.keys()})
            df_power_t = df_controlled.loc[df_controlled.index >= df_power_t.index[0], :].combine_first(base_power)
            df_t = {}

            for g in meter_dicts.keys():
                df_power_g = df_power_t.loc[:, [g]]
                df_power_g.rename(columns={g: 'total consumption'}, inplace=True)
                df_force_off_g = pd.DataFrame(0, index=pd.date_range(
                    t + time_specs.loc['force_off_long', 'start_time'],
                    t + time_specs.loc['force_off_long', 'end_time'] - pd.Timedelta('{}s'.format(dt)), freq='{}s'.format(dt)), columns=['force_off_long'])

                df_force_off_g.update(df_force_off[[g]].rename(columns={g: 'force_off_long'}))

                # df_force_off_g = df_force_off_g.combine_first(pd.DataFrame({'force_off_long': groups_controls[g].values}, index=pd.date_range(t, periods=groups_controls[g].values.shape[0], freq='{}s'.format(dt))))
                df_g = df_meteo_t.combine_first(df_power_g)
                df_g = df_g.combine_first(df_force_off_g)
                df_t[g] = df_g
            df_t = pd.concat(df_t, axis=1)

        df_t.fillna(0, inplace=True)


        price = get_day_ahead_price(t + pd.Timedelta('{}s'.format(dt)), t + pd.Timedelta('1D') + pd.Timedelta('{}s'.format(dt))) / 1000
        peak_price = 6.500

        # opt_vars = controller.get_opt_vars_names(x_flex_dict)
        # x_flex_baseline_dict = x_flex_dict.copy()
        # x_flex_baseline_dict[opt_vars] = 0
        # last_day_baseline_history = simulator.simulate(x_flex_baseline_dict)
        t1 = time.time()
        print('{} step {} took {}s to prepare data'.format(sim_name, t, t1-t0))
        if optimization_type =='sequential':
            opt_fun = controller.sequential_optimization
        elif optimization_type =='concurrent':
            opt_fun = controller.concurrent_optimization
        else:
            raise 'Unknown optimization_type'

        groups_controls, group_responses_oracle, p_hat_total = opt_fun(
            x_total=df_dso.loc[[t]],
            prices=price.values,
            peak_price=peak_price,
            dt=dt,
            last_peak=last_peaks[t.month],
            groups_dicts=meter_dicts,
            baseline_in_p_hat=baseline_in_p_hat,
            savedir=figs_dir,
            basename='{}'.format(t),
            formatter=oracle['model'].formatter,
            df=df_t, meter_dicts=meter_dicts, cfg=cfg_metadata, oracle_type='power', solver_pars=solver_pars)

        t2 = time.time()
        print('{} step {} took {}s to generate control'.format(sim_name, t, t2 - t1))

        # after the control has been decided, simulate one day of control
        if metamodel:
            group_responses = group_responses_oracle
        else:
            group_responses = sim_interface.simulate(t, df_meteo_t, meter_dicts, groups_controls)

        last_peaks[t.month] = np.maximum(last_peaks[t.month], p_hat_total.max(axis=1).values[0])
        groups_controls.index = pd.date_range(t + pd.Timedelta('{}s'.format(dt)), t + pd.Timedelta('1D'), freq='{}s'.format(dt))
        group_responses.index = pd.date_range(t + pd.Timedelta('{}s'.format(dt)), t + pd.Timedelta('1D'), freq='{}s'.format(dt))
        group_responses_oracle.index = pd.date_range(t + pd.Timedelta('{}s'.format(dt)), t + pd.Timedelta('1D'), freq='{}s'.format(dt))
        for g in meter_dicts.keys():
            group_responses[g, 'tot_oracle'] = group_responses_oracle[g, 'tot']
            group_responses[g, 'baseline'] = group_responses_oracle[g, 'baseline']
        group_responses_all = group_responses_all.combine_first(group_responses)
        df_force_off = df_force_off.combine_first(groups_controls)
        last_day_history = group_responses.loc[:, (slice(None), 'tot')].droplevel(1, axis=1)
        last_day_history_oracle = group_responses_oracle.loc[:, (slice(None), 'tot')].droplevel(1, axis=1)



        df_controlled = df_controlled.combine_first(last_day_history)
        df_controlled_oracle = df_controlled_oracle.combine_first(last_day_history_oracle)
        # df_baseline = df_controlled.combine_first(last_day_baseline_history)
        plt.close('all')

        if save_figs:
            t_fig = time.time()
            # import seaborn as sb
            # sb.set_style('darkgrid')
            fig, ax = plt.subplots(3, 1, figsize=(6, 4.5), sharex=True)
            ax2 = ax[0].twinx()
            df_force_off.loc[(df_force_off.index >= price.index[0]) & (df_force_off.index <= price.index[-1])].plot(linestyle='--', ax=ax2, legend=False)
            (df_controlled.loc[(df_controlled.index >= price.index[0]) & (df_controlled.index <= price.index[-1])]/1000).plot(ax=ax[0])
            (df_controlled_oracle.loc[(df_controlled_oracle.index >= price.index[0]) & (df_controlled_oracle.index <= price.index[-1])]/1000).plot(ax=ax[0], style=':')
            filt_df_dso_measured = (df_dso_measured.index >= price.index[0]) & (df_dso_measured.index <= price.index[-1])
            filt_group_responses_all = (group_responses_all.index >= price.index[0]) & (group_responses_all.index <= price.index[-1])
            if baseline_in_p_hat:
                ((df_dso_measured.loc[filt_df_dso_measured] - group_responses_all.loc[filt_group_responses_all, (slice(None), 'baseline')].sum(axis=1)) / 1000).plot(ax=ax[1], label='uncontrolled', legend=True)
                (df_dso_measured.loc[filt_df_dso_measured] / 1000).plot(ax=ax[1], label='baseline', legend=True)
                ((df_dso_measured.loc[filt_df_dso_measured] + group_responses_all.loc[filt_group_responses_all, (slice(None), 'diff')].sum(axis=1))/1000).plot(ax=ax[1], label='controlled', legend=True)
            else:
                (df_dso_measured.loc[filt_df_dso_measured] / 1000).plot(ax=ax[1], label='uncontrolled', legend=True)
                ((df_dso_measured.loc[filt_df_dso_measured] + group_responses_all.loc[filt_group_responses_all, (slice(None), 'baseline')].sum(axis=1)) / 1000).plot(ax=ax[1], label='baseline', legend=True)
                ((df_dso_measured.loc[filt_df_dso_measured] + group_responses_all.loc[filt_group_responses_all, (slice(None), 'tot')].sum(axis=1)) / 1000).plot(ax=ax[1], label='controlled', legend=True)
                if not metamodel:
                    ((df_dso_measured.loc[df_force_off.index] + group_responses_all.loc[filt_group_responses_all, (slice(None), 'tot_oracle')].sum(axis=1)) / 1000).plot(ax=ax[1], label='controlled oracle', legend=True)
            # price_to_plot = get_day_ahead_price(df_force_off.index[0], df_force_off.index[-1])
            # price_to_plot.loc[(price_to_plot.index >= price.index[0]) & (price_to_plot.index <= price.index[-1])].plot(ax=ax[2])
            price.plot(ax=ax[2])
            # get_day_ahead_price(df_force_off.index[0], df_force_off.index[-1]).plot(ax=ax[2])
            ax[2].set_xlabel('timestep [15 min]')
            ax[0].set_ylabel('power [MW]')
            ax[1].set_ylabel('power [MW]')
            ax2.set_ylabel('force off [-]')
            ax[2].set_ylabel('price [€/MWh]')
            if show_plots:
                plt.pause(5)
            # plt.savefig(join(figs_dir, '{}_from_start.pdf'.format(t)))
            ax[0].set_xlim(price.index[0], price.index[-1])
            # plot legend outside of plot
            ax[0].legend(loc='upper left', ncols=4, fontsize='xx-small', bbox_to_anchor=(-0.005, 1.37))
            ax[1].legend(loc='upper left', ncols=4, fontsize='xx-small', bbox_to_anchor=(-0.005, 1.24))
            ax[2].legend(loc='upper left', ncols=2, fontsize='xx-small', bbox_to_anchor=(-0.005, 1.24))
            plt.subplots_adjust(bottom=0.2, top=0.90, hspace=0.25)
            plt.savefig(join(figs_dir, '{}_day.pdf'.format(t)))
            print('saving figs at t:{} took {}s'.format(t, time.time()-t_fig))
        group_responses_all.to_pickle(join(pks_dir, 'group_responses_all_{}.zip'.format(t)))
        df_force_off.to_pickle(join(pks_dir, 'df_force_off_{}.zip'.format(t)))
        print('{} step {} took {}s'.format(sim_name, t, time.time()-t0))
    print('{} finished'.format(sim_name))