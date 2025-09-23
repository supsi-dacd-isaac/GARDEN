from os.path import join
import pandas as pd

data_dir = 'data/processed'
df = pd.read_pickle(join(data_dir, 'all_egids.zip'))


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import os
from io import StringIO
import time

# the following information is taken from the REA website https://www.regbl.admin.ch/catalog/it/4.2/GWAERZH2/final

mapping_dfs = {}

gba = {
    8011: "Before 1919",
    8012: "1919 - 1945",
    8013: "1946 - 1960",
    8014: "1961 - 1970",
    8015: "1971 - 1980",
    8016: "1981 - 1985",
    8017: "1986 - 1990",
    8018: "1991 - 1995",
    8019: "1996 - 2000",
    8020: "2001 - 2005",
    8021: "2006 - 2010",
    8022: "2011 - 2015",
    8023: "From 2016",
}
mapping_dfs['GBAUP'] = pd.DataFrame(list(gba.items()), columns=["Code", "Period of construction"])

gheat = {
    7500: "None",
    7501: "Air",
    7510: "Geothermal (generic)",
    7511: "Geothermal probe",
    7512: "Geothermal coil",
    7513: "Water (groundwater, surface water, wastewater)",
    7520: "Gas",
    7530: "Heating oil",
    7540: "Wood (generic)",
    7541: "Wood (logs)",
    7542: "Wood (pellets)",
    7543: "Wood (wood chips, shavings)",
    7550: "Waste heat (within the building)",
    7560: "Electricity",
    7570: "Sun (thermal)",
    7580: "District heat (generic)",
    7581: "District heat (high temperature)",
    7582: "District heat (low temperature)",
    7598: "Undetermined",
    7599: "Other",
}
mapping_dfs['GENH1'] = pd.DataFrame(list(gheat.items()), columns=["Code", "Heating source 1"])
mapping_dfs['GENH2'] = pd.DataFrame(list(gheat.items()), columns=["Code", "Heating source 2"])
mapping_dfs['GENW1'] = pd.DataFrame(list(gheat.items()), columns=["Code", "Hot water energy source 1"])
mapping_dfs['GENW2'] = pd.DataFrame(list(gheat.items()), columns=["Code", "Hot water energy source 2"])


ggen = {
    7400: "No heat generator",
    7410: "Heat pump for single building",
    7411: "Heat pump for multiple buildings",
    7420: "Solar thermal system for single building",
    7421: "Solar thermal system for multiple buildings",
    7430: "Boiler (generic) for single building",
    7431: "Boiler (generic) for multiple buildings",
    7432: "Standard boiler for single building",
    7433: "Standard boiler for multiple buildings",
    7434: "Condensing boiler for single building",
    7435: "Condensing boiler for multiple buildings",
    7436: "Stove",
    7440: "CHP plant for single building",
    7441: "CHP plant for multiple buildings",
    7450: "Electric central heating for single building",
    7451: "Electric central heating for multiple buildings",
    7452: "Electric radiators (including infrared)",
    7460: "District heating exchanger and system for single building",
    7461: "District heating exchanger and system for multiple buildings",
    7499: "Other",
}
mapping_dfs['GWAERZH1'] = pd.DataFrame(list(ggen.items()), columns=["Code", "Heat generator 1"])
mapping_dfs['GWAERZH2'] = pd.DataFrame(list(ggen.items()), columns=["Code", "Heat generator 2"])


gwatergen = {
    7600: "No heat generator",
    7610: "Heat pump (PAC)",
    7620: "Solar thermal system",
    7630: "Boiler (generic)",
    7632: "Standard boiler",
    7634: "Condensing boiler",
    7640: "CHP plant",
    7650: "Electric central boiler",
    7651: "Small boiler",
    7660: "District heating exchanger and system",
    7699: "Other",
}
mapping_dfs['GWAERZW1'] = pd.DataFrame(list(gwatergen.items()), columns=["Code", "Hot water heat generator 1"])
mapping_dfs['GWAERZW2'] = pd.DataFrame(list(gwatergen.items()), columns=["Code", "Hot water heat generator 2"])



def remap_column(mapping_dfs, df: pd.DataFrame, colname: str) -> pd.DataFrame:
    """Replace a coded column in a DataFrame with its description using a REA mapping file."""

    mapping_df = mapping_dfs[colname]
    # Determine mapping columns
    col_code = mapping_df.columns[0]
    col_desc = mapping_df.columns[1]

    # Build dictionary: {code: description}
    code_to_desc = dict(zip(mapping_df[col_code], mapping_df[col_desc]))

    # Rename column in new DataFrame
    df_new = df.copy()
    italian_name = col_desc  # usually the second column is the Italian label

    # Replace values and rename column
    df_new[italian_name] = df_new[colname].map(code_to_desc)
    df_new = df_new.drop(columns=[colname])

    print('code {} remapped to {}'.format(colname, italian_name))

    return df_new


for k in mapping_dfs:
    df = remap_column(mapping_dfs, df, k)
