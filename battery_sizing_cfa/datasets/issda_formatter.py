from os import listdir
from os.path import join
from tqdm import tqdm
import pandas as pd
import numpy as np


def ireland_data_formatting():
    base_path = 'datasets/ireland/CER Electricity Revised March 2012/'

    # get list of .txt files in base_path
    txt_files = np.sort([f for f in listdir(base_path) if f.endswith('.txt')])

    # read all the .txt files and concatenate them
    dfs_pivoted = []
    for txt_file in tqdm(txt_files):
        print('\nReading {}...'.format(txt_file))
        df = pd.read_csv(join(base_path, txt_file), parse_dates=False, index_col=1, sep=' ', header=None, names=['ID','time','power'])
        print('\nParsing time {}...'.format(txt_file))
        df.index = df.index.astype(str)
        df.index = (df.index.str[3:].astype(int).values*pd.Timedelta('30m') +
                    df.index.str[0:3].astype(int).values*pd.Timedelta('1D') +
                    pd.Timestamp('01-01-2009'))
        # some times are repeated, so we take the mean of the power
        print('\nPivoting {}...'.format(txt_file))
        dfs_pivoted.append(df.pivot_table(index=df.index, columns='ID', values='power', aggfunc='mean'))
        # remove IDs with too many missing values
        print('\nRemoving IDs with more than 5% of missing values...')

        dfs_pivoted[-1] = dfs_pivoted[-1].loc[:, dfs_pivoted[-1].isna().sum() < 0.05*dfs_pivoted[-1].shape[0]]


    df = pd.concat(dfs_pivoted, axis=1)
    df.ffill().bfill().to_pickle('datasets/ireland/power.pk')



if __name__ == '__main__':
    ireland_data_formatting()