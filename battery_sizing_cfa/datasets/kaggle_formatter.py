from os.path import join, exists, isdir
from tqdm import tqdm
import shutil
import zipfile
import pandas as pd
from os import remove
import matplotlib.pyplot as plt
from os import listdir

def keggle_data_formatting():
    base_path = 'datasets/london/'
    zip_filepath = join(base_path,'archive.zip')

    block_list = ['block_{}.csv'.format(i) for i in range(111)]
    # unzip all the files from data path which are in csv_list
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        for filename in tqdm(block_list):
            with zip_ref.open(join('halfhourly_dataset/halfhourly_dataset/', filename)) as zf, open(join(base_path, filename), 'wb') as f:
                shutil.copyfileobj(zf, f)

    filename = 'weather_hourly_darksky.csv'
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        with zip_ref.open(filename) as zf, open(join(base_path, filename), 'wb') as f:
            shutil.copyfileobj(zf, f)



    # read all the csv files, parse them, concat them and save them as a single compresed dataframe
    n_splits = 10
    files_per_split = len(block_list) // n_splits
    df_list = []
    k = 0
    # read weather data and keep only the float columns
    weather = pd.read_csv(join(base_path, 'weather_hourly_darksky.csv'), index_col=3, parse_dates=True, na_values=('Null', 'nan'))
    weather = weather.loc[:, weather.dtypes == 'float64']
    weather = weather.resample('30min').mean().ffill()
    for filename in tqdm(block_list):
        df = pd.read_csv(join(base_path, filename), index_col=1, parse_dates=True, na_values=('Null', 'nan'))
        df_list.append(df)
        if len(df_list) == files_per_split:
            df = pd.concat(df_list, axis=0)
            df_matrix = df.reset_index().pivot(index='tstp', columns='LCLid', values='energy(kWh/hh)')
            df_matrix.to_pickle(join(base_path, 'data_matrix_{}.zip'.format(k)))
            df_list = []
            k += 1

    # delete all the csv files in csv_list
    for filename in tqdm(block_list):
        remove(join(base_path, filename))

    # put everything in a single zip file
    matrices = []
    for f in tqdm(range(n_splits)):
        filename = 'data_matrix_{}.zip'.format(f)
        matrices.append(pd.read_pickle(join(base_path, filename)))
    df_matrix = pd.concat(matrices, axis=1)

    # filter out periods with few meters
    presence_ratio = (~df_matrix.isna()).sum(axis=1)/df_matrix.shape[1]
    presence_ratio.plot()
    plt.show()
    # keep only the periods with more than 80% of the meters
    df_matrix = df_matrix.loc[presence_ratio > 0.8]
    presence_ratio = (~df_matrix.isna()).sum(axis=1)/df_matrix.shape[1]
    presence_ratio.plot()
    plt.show()

    # filter out meters with too many missing values
    nan_ratio =  df_matrix.isna().sum().sort_values()/df_matrix.shape[0]
    (nan_ratio.cumsum()/nan_ratio.sum()).plot()
    plt.show()
    # keep only the meters with less than 5% of missing values
    df_matrix = df_matrix.loc[:, nan_ratio < 0.01]

    df_matrix = pd.concat(
        {'power':df_matrix, 'weather': weather.loc[(weather.index >= df_matrix.index[0]) & (weather.index <= df_matrix.index[-1])]},
        axis=1)

    df_matrix.to_pickle(join(base_path, 'data_matrix.zip'))

    # delete all the zip files
    for k in tqdm(range(n_splits)):
        remove(join(base_path, 'data_matrix_{}.zip'.format(k)))


if __name__ == '__main__':
    keggle_data_formatting()