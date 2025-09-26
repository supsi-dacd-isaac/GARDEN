# --------------------------------------------------------------------------------------------------------------------
# To download the datasets, we use the Kaggle API. To use it, you need to create an account on Kaggle and download
# your API credentials. You need to genrate a new token and download the kaggle.json file. Then, you need to move
# the file to the .kaggle folder in your home directory under ./kaggle.
# If the folder does not exist, you need to create it.
# You can find more information here: https://www.kaggle.com/docs/api
# --------------------------------------------------------------------------------------------------------------------

from os.path import join, exists
from os import mkdir
from urllib.request import urlretrieve
from zipfile import ZipFile
from os import listdir
import os
import pandas as pd
from battery_sizing_cfa.datasets.kaggle_formatter import keggle_data_formatting
from battery_sizing_cfa.datasets.issda_formatter import ireland_data_formatting

from urllib.parse import urlparse

# ---------------------------------- Download the Portugal and Rolle datasets ----------------------------------------

urls = {'portugal': 'https://github.com/TorchSpatiotemporal/multivariate-time-series-data/blob/master/electricity/electricity.txt.gz?raw=true',
        'rolle': 'https://zenodo.org/records/4549296/files/power_data.p?download=1'}

for name, url in urls.items():
    # make a directory with the name of the dataset, if not there
    data_dir = join('battery_sizing_cfa/datasets', name)
    if not exists(data_dir):
        mkdir(data_dir)

    path = urlparse(url).path
    ext = os.path.splitext(path)[1]
    name += ext


    print('Downloading {}...'.format(name))
    urlretrieve(url, join(data_dir, name))
    # get list of all files in data_dir
    files = listdir(data_dir)
    # unzip the files if needed
    for file in files:
        if file.endswith('.zip'):
            print('Unzipping {}...'.format(file))
            with ZipFile(join(data_dir, file), 'r') as zipObj:
                zipObj.extractall(data_dir)
    print('Done!')




# ---------------------------------- Format Portugal dataset ----------------------------------------

default_freq = '1h'
start_date = '01-01-2012 00:00'
raw_file_names='portugal.gz'
base_path = 'battery_sizing_cfa/datasets/portugal/'
df = pd.read_csv(join(base_path, raw_file_names),
                 index_col=False,
                 header=None,
                 sep=',',
                 compression='gzip')
index = pd.date_range(start=start_date,
                      periods=len(df),
                      freq=default_freq)
df = df.set_index(index)

df.to_pickle(join(base_path, 'portugal.pk'))


# ---------------------------------- Format London dataset ----------------------------------------
keggle_data_formatting()

# ---------------------------------- Format Ireland dataset ----------------------------------------
ireland_data_formatting()


# ---------------------------------- Save just the power time series for all the datasets -----------------------------

data_paths = {'portugal':'battery_sizing_cfa/datasets/portugal/portugal.pk',
              'rolle':'battery_sizing_cfa/datasets/rolle/rolle.p',
              'london':'battery_sizing_cfa/datasets/london/data_matrix.zip'}


pd.read_pickle(data_paths['portugal']).ffill().bfill().to_pickle('datasets/portugal/power.pk')
pd.read_pickle(data_paths['rolle'])['P_mean'].ffill().bfill().to_pickle('datasets/rolle/power.pk')
pd.read_pickle(data_paths['london'])['power'].ffill().bfill().to_pickle('datasets/london/power.pk')

