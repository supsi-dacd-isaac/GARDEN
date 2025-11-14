import numpy as np
from time import sleep
import os
from subprocess import Popen, STDOUT
from utils.loggers import setup_logger
import json
import pandas as pd
import pvlib
import glob
from multiprocessing import Pool, Semaphore
from copy import deepcopy


SEMAPHORE = Semaphore(1)


def launch_simulation(sim_confs, logger, python_venv, run_simgrid, run_mulder, wd):
    # logger.info('Starting sequence')
    cwd = os.getcwd()
    os.chdir(wd)
    sim_confs = np.sort(sim_confs)
    for sim_conf in sim_confs:
        algo_conf = sim_conf.replace('sim', 'algo').replace('grid_algo', 'grid_sim')
        sim_log = sim_conf.replace('json', 'log')
        algo_log = algo_conf.replace('json', 'log')

        if os.path.exists(sim_log):
            with open(sim_log, 'r') as f:
                f_content = f.read()
                if 'Simulation ended' in f_content and 'Simulation failed' not in f_content:
                    logger.info('skipping {}'.format(os.path.basename(sim_conf).split('.')[0]))
                    continue

        if os.path.exists(algo_conf):
            # logger.info('BG: %s -c %s -l %s' % (run_simgrid, sim_conf, sim_log))
            with open(sim_log, 'a') as log_file:
                with SEMAPHORE:
                    logger.info('Simulation {} started'.format(os.path.basename(sim_conf).split('.')[0]))
                    sim_proc = Popen([python_venv, run_simgrid, '-c', sim_conf, '-l', sim_log], stdout=log_file, stderr=log_file)
                    sleep(10)
            # logger.info('FG: %s -c %s -ct %s -l %s' % (run_mulder, sim_conf, algo_conf, algo_log))
            with open(algo_log, 'a') as log_file:
                algo_proc = Popen([python_venv, run_mulder, '-c', sim_conf, '-ct', algo_conf, '-l', algo_log],  stdout=log_file, stderr=log_file)
            algo_proc.communicate()
            sim_proc.communicate()
        else:
            # logger.info('FG: %s -c %s -l %s' % (run_simgrid, sim_conf, sim_log))
            with SEMAPHORE:
                logger.info('Simulation {} started'.format(os.path.basename(sim_conf).split('.')[0]))
                sim_proc = Popen([python_venv, run_simgrid, '-c', sim_conf, '-l', sim_log], stdout=open(sim_log, 'a'), stderr=STDOUT)
                sleep(10)
            sim_proc.communicate()
        logger.info('Simulation {} finished'.format(os.path.basename(sim_conf).split('.')[0]))
    os.chdir(cwd)
    # logger.info('Finished sequence')


class SimulatorInterface:
    def __init__(self, sim_name, sim_base_conf, df_client, dt, sim_interface_conf, logger=None, pool_size=4):
        self.logger = setup_logger('simulation') if logger is None else logger
        self.sim_name = sim_name
        self.df_client = df_client
        self.dt = pd.Timedelta(seconds=dt)
        self.t_offset = pd.Timedelta('{}D'.format((9*365+2)//7*7))
        self.dt_sim = pd.Timedelta(seconds=300)
        self.sim_base_conf = json.load(open(sim_base_conf))
        self.sim_base_conf['meteo']['station'] = 'garden_{}'.format(self.sim_name)
        self.sim_base_conf['meteo']['tags']['name'] = 'garden_{}'.format(self.sim_name)
        self.sim_base_conf['simulation']['ID'] = 'garden_{}'.format(self.sim_name)
        self.sim_base_conf['simulation']['internalID'] = ''
        self.sim_base_conf['simulation']['name'] = 'garden_{}'.format(self.sim_name)
        self.confs_base_path = os.path.join(sim_interface_conf['confs_base_path'], sim_name)
        self.sim_interface_conf = sim_interface_conf
        self.pool_size = pool_size

    def split_conf(self, base_conf):
        max_meters_per_sim = 96
        meters = base_conf['meters']
        num_meters = len(meters)
        num_sims = np.ceil(num_meters / max_meters_per_sim).astype(int)
        starting_meter = np.round(np.linspace(0, num_meters, num_sims + 1)).astype(int)
        split_confs = dict()
        for i in range(num_sims):
            start_ix, end_ix = starting_meter[i], starting_meter[i + 1]
            self.logger.debug('start {} end {}'.format(start_ix, end_ix))
            split_conf = deepcopy(base_conf)
            split_conf['meters'] = meters[start_ix:end_ix]
            meter_names = [m['name'] for m in split_conf['meters']]
            for k in split_conf['simulation']['internalController']['componentsProfiles'].keys():
                split_conf['simulation']['internalController']['componentsProfiles'][k] = {m: fo for m, fo in split_conf['simulation']['internalController']['componentsProfiles'][k].items() if m in meter_names}
            split_sim_name = '{:04d}_{:04d}'.format(start_ix, end_ix-1)
            split_conf['simulation']['internalID'] = split_conf['simulation']['ID'] + '_' + split_sim_name
            split_confs[split_sim_name] = split_conf
        return split_confs

    def write_confs(self, t, meter_dicts, groups_controls):
        conf = self.sim_base_conf.copy()
        conf = json.loads(json.dumps(conf).replace('"send_interval": 1', '"send_interval": 288'))
        conf['simulation']['internalController']['componentsProfiles'] = {'heat_pump': {}, 'electric_heater': {}}
        for group_name, group in meter_dicts.items():
            conf['simulation']['internalController']['componentsProfiles'][group['type']].update({meter: 'garden_{}_{}'.format(self.sim_name, group_name) for meter in group['meters']})

        conf['simulation']['startDateTime'] = (t+self.dt + self.t_offset).strftime('%Y-%m-%d %H:%M')
        conf['simulation']['startDateTimeMeteo'] = (t + self.dt + self.t_offset).strftime('%Y-%m-%d %H:%M')
        conf['simulation']['endDateTime'] = (t + self.dt + pd.Timedelta('1D') + self.t_offset).strftime('%Y-%m-%d %H:%M')
        split_confs = self.split_conf(conf)
        sim_dir = os.path.join(self.confs_base_path, (t + self.dt).strftime('%Y-%m-%d'), 'sim')
        os.makedirs(sim_dir, exist_ok=True)
        for conf_name, split_conf in split_confs.items():
            self.logger.info('saving {}'.format(conf_name))
            with open(os.path.join(sim_dir, '{}.json'.format(conf_name)), 'w') as f:
                json.dump(split_conf, f, indent=4)

    def write_force_off(self, t, group_controls):
        for group_name in group_controls.columns:
            df = pd.DataFrame(index=pd.date_range(start=t+self.dt, periods=len(group_controls), freq=self.dt), data={'ForceOff': group_controls[group_name].values})
            new_index = pd.date_range(start=t+self.dt, periods=86400/self.dt_sim.seconds, freq=self.dt_sim)
            up_df = df.reindex(new_index).interpolate(method='ffill')
            up_df.index += self.t_offset
            self.df_client.write_points(up_df, measurement='input', tags={'type': 'ForceOff', 'name': 'garden_{}_{}'.format(self.sim_name, group_name)})

    def write_meteo(self, t, df_meteo_t):
        location = pvlib.location.Location(latitude=float(self.sim_base_conf['location']['latitude']),
                                           longitude=float(self.sim_base_conf['location']['longitude']),
                                           altitude=float(self.sim_base_conf['location']['altitude']),
                                           tz=self.sim_base_conf['location']['time_zone'])
        index = pd.date_range(start=t + self.dt, periods=86400 / self.dt_sim.seconds, freq=self.dt_sim, tz='UTC')
        ghi = df_meteo_t['mean_GHI']
        ghi = ghi.reindex(index).interpolate(method='pchip')
        t_amb = df_meteo_t['mean_T']
        t_amb = t_amb.reindex(index).interpolate(method='pchip')
        sol_pos = location.get_solarposition(index)
        dni = pvlib.irradiance.disc(ghi, sol_pos['zenith'], ghi.index)['dni']
        # dni = pvlib.irradiance.dirint(ghi.values*10, sol_pos['zenith'].values, ghi.index)

        dhi = ghi - np.cos(sol_pos['zenith'] * np.pi / 180) * dni
        dni_extra = pvlib.irradiance.get_extra_radiation(index)
        meteo_df = pd.DataFrame(index=index, data={'GHI': ghi, 'DNI': dni, 'DHI': dhi, 'T': t_amb})
        tilt = 90
        azimuths = [0, 90, 180, 270]
        g_names = ['G_N', 'G_E', 'G_S', 'G_W']
        for name, azimuth in zip(g_names, azimuths):
            meteo_df[name] = pvlib.irradiance.get_total_irradiance(tilt, azimuth,
                                                                 sol_pos["zenith"],
                                                                 sol_pos["azimuth"],
                                                                 dni,
                                                                 ghi,
                                                                 dhi,
                                                                 dni_extra=dni_extra,
                                                                 model='haydavies')['poa_global']
        meteo_df['G_F'] = meteo_df[g_names].mean(axis=1)
        meteo_df.index += self.t_offset
        self.df_client.write_points(meteo_df, measurement='input', tags={'type': 'meteo', 'name': 'garden_{}'.format(self.sim_name)})

    def run_simulation(self, t):
        self.logger.info('Starting simulation')
        sim_dir = os.path.join(self.confs_base_path, (t + self.dt).strftime('%Y-%m-%d'), 'sim')
        conf_files = glob.glob(os.path.join(sim_dir, '*.json'))
        sims = [([s], self.logger, self.sim_interface_conf['python_venv'], self.sim_interface_conf['run_simgrid'],
                 self.sim_interface_conf['run_mulder'], self.sim_interface_conf['wd']) for s in conf_files]
        with Pool(self.pool_size) as p:
            p.starmap(launch_simulation, sims)

    def retrieve_simulation_results(self, t, meter_dicts):
        measurement = 'sim_results_models_garden_{}'.format(self.sim_name)
        t_start = round((t+self.dt + self.t_offset).timestamp())
        t_end = round((t + self.dt + pd.Timedelta('1D') + self.t_offset).timestamp())
        group_responses_df = pd.DataFrame(columns=pd.MultiIndex.from_product([meter_dicts.keys(), ['tot']]))
        for meter_dict_name, meter_dict in meter_dicts.items():
            meter_tags = '('
            for meter in meter_dict['meters']:
                meter_tags += '"meter_id" = \'{}\' OR '.format(meter)
            meter_tags = meter_tags[:-4]+')'
            query = 'SELECT {}*mean("value") FROM {} WHERE ({} AND "signal"= \'p_tot\' AND "component_type" = \'{}\') AND time >= {}s AND time< {}s GROUP BY time({}s)'.format(len(meter_dict['meters']), measurement, meter_tags, meter_dict['type'], t_start, t_end, self.dt.seconds)
            df = self.df_client.query(query)[measurement]
            group_responses_df[meter_dict_name, 'tot'] = df['mean']
        group_responses_df.index = np.arange(0, len(group_responses_df.index))
        return group_responses_df

    def simulate(self, t, df_meteo_t, meter_dicts, groups_controls):
        self.write_force_off(t, groups_controls)
        self.write_meteo(t, df_meteo_t)
        self.write_confs(t, meter_dicts, groups_controls)
        self.run_simulation(t)
        return self.retrieve_simulation_results(t, meter_dicts)
