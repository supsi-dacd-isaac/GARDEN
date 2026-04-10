import random
import json
import os
import geopandas as gpd
import requests
from os.path import join
import tempfile
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import contextily as cx
from typing import Optional

COL_YEAR="GBAUJ"
COL_GBAUP="GBAUP"

def s_int(x):  return None if pd.isna(x) else int(round(float(x)))
def s_float(x): return None if pd.isna(x) else float(x)

def cp_from_year(y):
    y = s_int(y)
    if y is None: return None
    if y <= 1919: return "<1918"
    if y <= 1948: return "1918-1948"
    if y <= 1980: return "1949-1980"
    if y <= 1994: return "1981-1994"
    if y <= 2000: return "1995-2000"
    if y <= 2010: return "2001-2010"
    return ">2011"

def map_construction_period(c:int):
    c = s_int(c)
    if c == 8011: return "<1918"
    if c == 8012: return "1918-1948"
    if c in (8013,8014,8015): return "1949-1980"
    if c in (8016,8017,8018): return "1981-1994"
    if c == 8019: return "1995-2000"
    if c in (8020,8021): return "2001-2010"
    if c in (8022,8023): return ">2011"
    return None

def construction_period(r):
    return cp_from_year(r[COL_YEAR]) or map_construction_period(r[COL_GBAUP])


def map_hot_water_system(genw1: int, gwaerzw1: int) -> str:
    """
    Map hot water system codes to system type string.
    
    Args:
        genw1: Energy type code (GENW1) from the building registry.
        gwaerzw1: System type code (GWAERZW1) from the building registry.
        
    Returns:
        Human-readable hot water system type string.
    """
    # GWAERZW1 - system type
    if gwaerzw1 == 7610:
        return "hp"
    elif gwaerzw1 == 7632:
        return "gas-boiler"  # non-condensing boiler
    elif gwaerzw1 == 7634:
        if genw1 == 7520:
            return "gas-condensing-boiler"
        elif genw1 == 7530:
            return "oil-condensing-boiler"
        return ""
    elif gwaerzw1 in (7650, 7651):
        return "electric-boiler"
    
    # Fallback with GENW1 - energy type
    if genw1 == 7560:
        return "electric-boiler"
    elif genw1 == 7520:
        return "gas-boiler"
    elif genw1 == 7530:
        return "oil-boiler"
    
    return ""


def map_heating_system(gwaerzh1: Optional[int] = None, genh1: Optional[int] = None) -> str:
    """
    Map heating system codes to system type string.
    
    Args:
        gwaerzh1: Heating system type code (GWAERZH1) from the building registry.
        genh1: Heating energy type code (GENH1) from the building registry.
        
    Returns:
        Human-readable heating system type string.
    """
    # Heat pump
    if gwaerzh1 in (7410, 7411):
        return "hp"
    
    # Electric heating
    if gwaerzh1 in (7450, 7451, 7452, 7436):
        return "electric-boiler"
    
    # Gas/Oil boiler (classic)
    if gwaerzh1 in (7430, 7431, 7432, 7436):
        if genh1 == 7520:
            return "gas-boiler"
        elif genh1 == 7530:
            return "oil-boiler"
    
    # Gas/Oil condensing boiler
    if gwaerzh1 == 7434:
        if genh1 == 7520:
            return "gas-condensing-boiler"
        elif genh1 == 7530:
            return "oil-condensing-boiler"
    
    return ""


def download_ticino_data():

    url = "https://public.madd.bfs.admin.ch/ti.zip"
    response = requests.get(url)
    with open('ti.zip', 'wb') as out_file:
        out_file.write(response.content)

    # Step 2: Extract 'buildings.geojson' from the zip file and load it with GeoPandas
    with tempfile.TemporaryDirectory() as tmpdirname:
        with zipfile.ZipFile('ti.zip', 'r') as zip_ref:
            zip_ref.extract('buildings.geojson', path=tmpdirname)
            zip_ref.extract('gebaeude_batiment_edificio.csv', path=tmpdirname)

        geojson_path = join(tmpdirname, 'buildings.geojson')
        bldg_raw_gdp = gpd.read_file(geojson_path)
        bldg_raw_gdp = bldg_raw_gdp.rename(columns={'egid': 'EGID'})
        csv_path = join(tmpdirname, 'gebaeude_batiment_edificio.csv')
        bldg_raw = pd.read_csv(csv_path, delimiter='\t', low_memory=False)

    df_merged = bldg_raw.merge(bldg_raw_gdp, on='EGID', how='left')
    df_merged = gpd.GeoDataFrame(df_merged, geometry='geometry', crs=bldg_raw_gdp.crs)

    return df_merged

def select_polygon(selection_polygon=None):
    # download the data if not already downloaded
    df_merged = download_ticino_data()
    # filter the egids that are inside the selection polygon
    if selection_polygon is not None:
        df_merged = df_merged[df_merged['geometry'].intersects(selection_polygon)]
    else:
        df_merged = df_merged

    # plot the selected polygons
    sample_gdp = df_merged.sample(frac=0.1, random_state=42)

    # Ensure the GeoDataFrame has a projected CRS for contextily
    # If it's not already projected, reproject it to Web Mercator (EPSG:3857)
    if sample_gdp.crs != 'EPSG:3857':
        sample_gdp = sample_gdp.to_crs(epsg=3857)

    # Plot the 'geometry' column
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sample_gdp.plot(ax=ax, color='blue', edgecolor='black', alpha=0.7)

    # Add OpenStreetMap basemap
    cx.add_basemap(ax, crs=sample_gdp.crs.to_string())

    ax.set_title('Random 10% Sample of Building Geometries with OSM Basemap')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.show()
    return df_merged



def generate_template(building_info):
    building_types = ["multi-family", "single-family"]
    building_type = random.choice(building_types)
    morphologies = ["1", "2", "3", "4"]
    orientations = ["north", "south", "east", "west"]
    construction_periods = ["1918", "1918-1948", "1949-1980", "1981-1994", "1995-2000", "2001-2010", "2011"]
    renovation_periods = ["noRenovation"] + construction_periods
    
    heating_fuels = ["Diesel", "NaturalGas", "Electricity"]
    heating_emitters = ["radiantFloor", "radiators"]
    heating_systems = ["boiler", "hp"]
    at_home_options = ["rarely", "often"]

    total_floors =  random.randint(1, 3 if building_type == "single-family" else 5)
    building_floor_area = building_info['GAREA']
    building_area = building_info['GAREA'] * total_floors

    HS_MAP = {
        "boiler": "boiler",
        "hp": "hp",
        'electric-boiler': 'boiler',
        'gas-boiler': 'boiler',
        'oil-boiler': 'boiler',
        'gas-condensing-boiler': 'boiler',
        'oil-condensing-boiler': 'boiler',

    }

    heating_system = HS_MAP[building_info['heating_system']] if building_info['heating_system'] in HS_MAP else random.choice(heating_systems)
    dhw_system = HS_MAP[building_info['hot_water_system']] if building_info['hot_water_system'] in HS_MAP else random.choice(heating_systems)
    construction_period = building_info['construction_period']
    building_floor_area = building_info['GAREA']



    template = {
        "buildingType": building_type,
        "totalFloors": total_floors,
        "buildingMorphology": random.choice(morphologies),
        "buildingOrientation": random.choice(orientations),
        "constructionPeriod": construction_period,
        "wallsRenovationPeriod": random.choice(renovation_periods),
        "floorsRenovationPeriod": random.choice(renovation_periods),
        "roofRenovationPeriod": random.choice(renovation_periods),
        "windowsRenovationPeriod": random.choice(renovation_periods),
        "buildingFloorArea": building_floor_area,
        "dwellingNumber": 1 if building_type == "single-family" else 2 * total_floors,
        "dwellingArea": 100, #FIXME
        "dwellingResidents": random.randint(1, 5),
        "shSetpoint": random.randint(18, 24),
        "shSystem": heating_system,
        "shFuel": random.choice(heating_fuels),
        "emitter": random.choice(heating_emitters),
        "shVolume": random.choice([0, random.randint(50, 500)]), #FIXME
        #"solarThermal": random.choice([True, False]),
        # "solarCollectorArea": random.randint(0, 20),
        # "solarTankVolume": random.randint(0, 500),
        "dhwSystem": dhw_system,
        "dhwFuel": random.choice(heating_fuels),
        "dhwVolume": 5, #FIXME
        "dhwSetpoint": 50, #FIXME
        "dhwSolarThermal": random.choice([True, False]),
        "dhwSolarThermalArea": 5, #FIXME
        "pvsystem": random.choice([True, False]),
        "pvCapacityKwp": random.randint(0, 20),
        "batteryCapacityKwh": random.choice([0, random.randint(1, 20)]),
        "evHomeCharging": random.choice([True, False]),
        "evKmYear": random.randint(1000, 30000),
        "atHome": random.choice(at_home_options),
        "annualConsumptionKwh": 0, #FIXME
        "includeFloorTemps": False,
        "showAllVars": False,

        "installPvPanels": random.choice([True, False]),
        "pvCapacityKwpNew": random.randint(0, 20),
        "installElectricBattery": random.choice([True, False]),
        "installEV": random.choice([True, False]),
        "evKmYearNew": random.randint(1000, 30000),
        "atHomeNew": random.choice(at_home_options),
        "retrofitWalls": random.choice([True, False]),
        "retrofitFloor": random.choice([True, False]),
        "retrofitRoof": random.choice([True, False]),
        "retrofitWindows": random.choice([True, False]),
        "DRstart": "17:00",
        "DRend": "19:00",
        "reducePeakSetpoint": random.randint(0, 2),
        "reduceGlobalSetpoint": random.randint(0, 2),
        "increasePreheatingSetpoint": random.randint(0, 2),
        "increaseTankTemperature": random.randint(0, 5),
        "installHeatPump": random.choice([True, False]),
        "installDhwSolarThermal": random.choice([True, False]),
        "flexibleAppliances": random.choice([True, False]),
        "flexibleEvCharging": random.choice([True, False]),
        "flexibleHeating": False, #FIXME
        "buildingArea": building_area,

    }

    return template


def main():
    n_max = 10
    k = 0
    df_merged = select_polygon()
    # map the building_info to the template
    #df_merged['construction_period'] = df_merged['GBAUP'].apply(map_construction_period)
    df_merged['heating_system'] = df_merged.apply(lambda row: map_heating_system(row['GWAERZH1'], row['GENH1']), axis=1)
    df_merged['hot_water_system'] = df_merged.apply(lambda row: map_hot_water_system(row['GENW1'], row['GWAERZW1']), axis=1)    
    df_merged['construction_period'] = df_merged.apply(construction_period, axis=1)
    path = "building_simulation/input_files/automatically_generated"
    os.makedirs(path, exist_ok=True)
    for i, row in df_merged.iterrows():
        if k >= n_max:
            break
        k += 1
        building_info = row.to_dict()
        template = generate_template(building_info)
        filename = f"test_{i}.json"
        with open(os.path.join(path, filename), "w") as f:
            json.dump(template, f, indent=2)
        print(f"✅ Saved {os.path.join(path, filename)}")


if __name__ == "__main__":
    # TODO: VERIFY THIS FILEEEEEEEE (with Manuel)
    main()