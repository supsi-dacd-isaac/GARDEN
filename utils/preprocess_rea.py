import numpy as np
import json
from os.path import join
import os
import geopandas as gpd
import pandas as pd
import requests
from tqdm import tqdm
import zipfile
import tempfile
import matplotlib.pyplot as plt
from utils.loggers import setup_logger

tqdm.pandas()


def installable_power_ratio(beta_ref_deg, beta_new_deg, phi_deg):
    """
    Computes the ratio of installable PV power when changing from
    a reference tilt angle (beta_ref_deg) to a new tilt angle (beta_new_deg),
    given a minimum winter sun elevation phi_deg (in degrees).

    The formula is based on spacing:
        S(beta) = cos(beta) + sin(beta)/tan(phi)
    (with panel depth L factored out, since it cancels in the ratio).

    Parameters:
        beta_ref_deg (float): Reference tilt angle in degrees.
        beta_new_deg (float): New tilt angle in degrees.
        phi_deg (float): Minimum winter sun elevation angle in degrees.

    Returns:
        float: Power ratio P(beta_new) / P(beta_ref).
               Multiply known P_ref by this ratio to get P_new.
    """
    # Convert degrees to radians
    beta_ref = np.radians(beta_ref_deg)
    beta_new = np.radians(beta_new_deg)
    phi = np.radians(phi_deg)

    # Spacing for reference tilt
    S_ref = np.cos(beta_ref) + (np.sin(beta_ref) / np.tan(phi))

    # Spacing for new tilt
    S_new = np.cos(beta_new) + (np.sin(beta_new) / np.tan(phi))

    # Power is inversely proportional to spacing
    # ratio = S_ref / S_new
    return S_ref / S_new


def select_column_REA(x):
    """
    Selects specific columns from the Real Estate Assessment (REA) dataset.

    Parameters:
        x (str): Column name to check.

    Returns:
        bool: True if column name is in the list of selected columns, False otherwise.
    """
    return x in ['EGID', 'GGDENAME', 'GKODE', 'GKODN']


def get_building_id(coords):
    """
    Retrieves the building ID for a given set of coordinates using the Swiss geo.admin.ch API.

    This function sends a request to the Swiss federal geoportal API to identify
    buildings at the specified coordinates, focusing on solar energy suitability data.

    Parameters:
        coords (str): Coordinates in the Swiss coordinate system (EPSG:2056).

    Returns:
        pandas.DataFrame: DataFrame containing the building ID and related attributes,
                         or a DataFrame with placeholder values if no building is found.
    """
    url = 'https://api3.geo.admin.ch/rest/services/api/MapServer/identify?'
    params = dict(
        lang='en',
        sr='2056',
        geometryType='esriGeometryPoint',
        geometry=coords,
        imageDisplay='0,0,0',
        mapExtent='0,0,0,0',
        tolerance='0',
        limit=10,
        returnGeometry='false',
        layers='all:ch.bfe.solarenergie-eignung-daecher'
    )

    res = requests.get(url, params=params)
    temp = json.loads(res.text)
    temp_2 = pd.DataFrame.from_dict(temp, orient='columns')
    temp_2.reset_index()
    bldg_PV_data = pd.DataFrame.from_dict(temp_2.iat[0, 0], orient='index')
    bldg_PV_data = bldg_PV_data.rename(columns={0: 'results'})

    if bldg_PV_data.loc['featureId', 'results'] == -99:
        data_fill = [-99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99]
        PV_POT_attr_filter = pd.DataFrame(data_fill).T
        PV_POT_attr_filter.columns = ['ausrichtung', 'neigung', 'df_nummer', 'gstrahlung', 'gwr_egid', 'building_id',
                                      'klasse', 'flaeche', 'label', 'stromertrag_sommerhalbjahr',
                                      'stromertrag_winterhalbjahr']
        PV_POT_attr_filter = PV_POT_attr_filter.rename(columns={
            'ausrichtung': 'AUSRICHTUNG', 'neigung': 'NEIGUNG', 'df_nummer': 'DF_NUMMER', 'label': 'DF_UID',
            'gstrahlung': 'GSTRAHLUNG', 'gwr_egid': 'GWR_EGID',
            'klasse': 'KLASSE', 'flaeche': 'FLAECHE', 'stromertrag_sommerhalbjahr': 'STROMERTRAG_SOMMERHALBJAHR',
            'stromertrag_winterhalbjahr': 'STROMERTRAG_WINTERHALBJAHR'})

    else:
        attributes = bldg_PV_data.loc['attributes'].to_dict()
        PV_POT_attr = pd.DataFrame.from_dict(attributes).T
        PV_POT_attr_filter = PV_POT_attr[['building_id']]

    return PV_POT_attr_filter


def get_PV_pot_bldgid(building_id):
    """
    Retrieves detailed photovoltaic potential data for a specific building ID.

    This function queries the Swiss federal geoportal API to find all roof sections
    of a building and their solar energy potential data.

    Parameters:
        building_id (str): Building identifier in the Swiss solar cadastre.

    Returns:
        pandas.DataFrame: DataFrame containing detailed PV potential data for all
                         roof sections of the specified building.
    """
    TOT_DATA = pd.DataFrame()
    url = 'https://api3.geo.admin.ch//rest/services/api/MapServer/find?'
    params = dict(
        layer='ch.bfe.solarenergie-eignung-daecher',
        searchField='building_id',
        searchText=building_id,
        contains='false'
    )

    res = requests.get(url, params=params)
    temp = json.loads(res.text)
    temp_3 = pd.DataFrame.from_dict(temp, orient='index')
    for i in range(len(temp_3.T)):
        POT_bldg_PV_data = pd.DataFrame.from_dict(temp_3.iat[0, i], orient='index')
        POT_attributes = POT_bldg_PV_data.loc['attributes'].to_dict()
        POT_attr = pd.DataFrame.from_dict(POT_attributes).T
        POT_attr = POT_attr[['ausrichtung', 'neigung', 'df_nummer', 'gstrahlung', 'gwr_egid', 'building_id',
                             'klasse', 'flaeche', 'label', 'stromertrag_sommerhalbjahr', 'stromertrag_winterhalbjahr']]
        POT_attr = POT_attr.rename(columns={
            'ausrichtung': 'AUSRICHTUNG', 'neigung': 'NEIGUNG', 'df_nummer': 'DF_NUMMER', 'label': 'DF_UID',
            'gstrahlung': 'GSTRAHLUNG', 'gwr_egid': 'GWR_EGID',
            'klasse': 'KLASSE', 'flaeche': 'FLAECHE', 'stromertrag_sommerhalbjahr': 'STROMERTRAG_SOMMERHALBJAHR',
            'stromertrag_winterhalbjahr': 'STROMERTRAG_WINTERHALBJAHR'})
        POT_attr = POT_attr.set_index('DF_UID')
        TOT_DATA = pd.concat([TOT_DATA, POT_attr])

    return TOT_DATA


def get_flat_roof_best_orientation(geom):
    """
    Determines the optimal azimuth orientation for solar panels on a flat roof.

    This function calculates the minimum rotated rectangle of the roof geometry,
    identifies the longest side, and determines the best orientation for solar panels
    based on the geometry's shape and dimensions.

    Parameters:
        geom (shapely.geometry): Shapely geometry object representing a roof.

    Returns:
        float: Optimal azimuth angle in degrees (0-360) for solar panel orientation.
    """
    # Calculate the oriented minimum bounding box
    oriented_bbox = geom.minimum_rotated_rectangle

    # Extract the coordinates of the bounding box
    bbox_coords = oriented_bbox.exterior.coords.xy
    bbox_points = np.array(list(zip(bbox_coords[0], bbox_coords[1])))

    # Calculate the lengths of the sides
    side_lengths = np.sqrt(np.sum(np.diff(bbox_points, axis=0) ** 2, axis=1))

    # Identify the longest side
    longest_side_idx = np.argmax(side_lengths)
    shortest_side_len = np.min(side_lengths)

    # Calculate the angle of the longest side
    dx = bbox_points[longest_side_idx + 1][0] - bbox_points[longest_side_idx][0]
    dy = bbox_points[longest_side_idx + 1][1] - bbox_points[longest_side_idx][1]
    angle = np.degrees(np.arctan2(dy, dx)) % 180
    if angle <= 90:
        azimuth = 180 - angle
    else:
        azimuth = 360 - angle

    if azimuth < 135 and shortest_side_len > 5:
        azimuth += 90
    elif azimuth > 225 and shortest_side_len > 5:
        azimuth -= 90
    # print(f'Azimuth: {azimuth}, Angle: {angle}')

    # fig, ax = plt.subplots()
    # gpd.GeoSeries(oriented_bbox).plot(ax=ax)
    # gpd.GeoSeries(geom).plot(ax=ax, color='red')
    # #plot a 10m arrow from the centroid in the azimuth direction
    # centroid = geom.centroid
    # ax.arrow(centroid.x, centroid.y, 10*np.cos(np.radians(90-azimuth)), 10*np.sin(np.radians(90-azimuth)), head_width=1, head_length=1, fc='k', ec='k')
    # plt.show()
    return azimuth


def preprocess_pv_pot_AEM(pv_pot_AEM, scenario='standard'):
    """
    Preprocesses the PV potential data for simulation.

    This function prepares the PV potential data by computing centroids, transforming coordinates,
    handling azimuth and tilt angles based on different scenarios, and calculating baseline
    installation parameters.

    Parameters:
        pv_pot_AEM (GeoDataFrame): GeoDataFrame containing PV potential data.
        scenario (str): Scenario name - 'standard', 'winter_soft' 'winter_aggressive', or 'winter_extreme',
                       which determines tilt angle distributions.

    Returns:
        GeoDataFrame: Preprocessed GeoDataFrame with additional columns for simulation.
    """
    # Compute centroids and transform coordinates
    pv_pot_AEM['centroid'] = pv_pot_AEM['geometry'].centroid
    pv_pot_AEM['centroid_x'] = pv_pot_AEM['centroid'].x
    pv_pot_AEM['centroid_y'] = pv_pot_AEM['centroid'].y
    pv_pot_AEM['point_4326'] = gpd.GeoSeries(pv_pot_AEM['centroid'], crs='EPSG:2056').to_crs('EPSG:4326')
    pv_pot_AEM['point_4326_x'] = pv_pot_AEM['point_4326'].x.round(5)  # round to 5 decimal places
    pv_pot_AEM['point_4326_y'] = pv_pot_AEM['point_4326'].y.round(5)

    # Handle azimuth and tilt adjustments
    pv_pot_AEM['azimuth'] = pv_pot_AEM['AUSRICHTUNG'] % 360.0
    flat_roofs = pv_pot_AEM['NEIGUNG'] == 0

    pv_pot_AEM['tilt'] = pv_pot_AEM['NEIGUNG']
    if scenario == 'standard':
        tilt_angles = np.array([0, 5, 10, 15, 20, 25, 30, 35])
        probabilities = np.array([0.01, 0.14, 0.35, 0.35, 0.10, 0.03, 0.015, 0.005])
    elif scenario == 'winter_soft':
        tilt_angles = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
        probabilities = np.array([0.03, 0.07, 0.15, 0.20, 0.18, 0.15, 0.07, 0.05, 0.04, 0.03, 0.02, 0.01])
    elif scenario == 'winter_aggressive':
        tilt_angles = np.array([50, 55, 60, 65, 70, 75])
        probabilities = np.array([0.05, 0.10, 0.30, 0.35, 0.15, 0.05])
    elif scenario == 'winter_extreme':
        tilt_angles = np.array([75, 80, 85, 90])
        probabilities = np.array([0.25, 0.25, 0.25, 0.25])

    tilts_on_flat_roofs = np.random.choice(tilt_angles, size=flat_roofs.sum(), p=probabilities)
    pv_pot_AEM.loc[flat_roofs, 'tilt'] = tilts_on_flat_roofs

    # Define discrete tilt angles (in degrees) for east-west configurations, excluding 0°
    tilt_angles_eastwest = np.array([5, 10, 15])
    probabilities_eastwest = np.array([0.25, 0.55, 0.20])
    tilts_on_flat_roofs_eastwest = np.random.choice(tilt_angles_eastwest, size=flat_roofs.sum(),
                                                    p=probabilities_eastwest)
    pv_pot_AEM.loc[flat_roofs, 'tilt_east_west'] = tilts_on_flat_roofs_eastwest

    # For flat roofs, compute the best orientation
    pv_pot_AEM.loc[flat_roofs, 'azimuth'] = pv_pot_AEM.loc[flat_roofs, 'geometry'].apply(
        get_flat_roof_best_orientation)

    # Compute kWp_installed and kWh_per_kWp
    pv_pot_AEM['kWp_installed_default'] = pv_pot_AEM['FLAECHE'] / 5.0
    pv_pot_AEM['kWh_per_kWp_default'] = pv_pot_AEM['STROMERTRAG'] / pv_pot_AEM['kWp_installed_default']
    pv_pot_AEM['kWh_per_kWp_winter'] = pv_pot_AEM['STROMERTRAG_WINTERHALBJAHR'] / pv_pot_AEM['kWp_installed_default']

    # Apply installable_power_ratio only to flat roofs
    mask = pv_pot_AEM['NEIGUNG'] == 0
    pv_pot_AEM.loc[mask, 'kWp_installed_default'] = installable_power_ratio(15, pv_pot_AEM.loc[mask, 'tilt'], 20) * \
                                                    pv_pot_AEM.loc[mask, 'kWp_installed_default']
    pv_pot_AEM.loc[mask, 'kWp_installed_east_west'] = pv_pot_AEM.loc[mask, 'kWp_installed_default'] * 1.7
    pv_pot_AEM.loc[mask, 'kWp_installed_east'] = pv_pot_AEM.loc[mask, 'kWp_installed_default']
    pv_pot_AEM.loc[mask, 'kWp_installed_west'] = pv_pot_AEM.loc[mask, 'kWp_installed_default']
    return pv_pot_AEM


def load_roofs(data_dir='pv_sim/data'):
    """
    Loads and prepares roof data from geospatial files.

    This function reads roof geometry and solar potential data from GPKG files,
    merges them, and adjusts values like area and energy production based on
    area intersection proportions for buildings with multiple roof sections.

    Parameters:
        data_dir (str): Directory containing the input data files.

    Returns:
        GeoDataFrame: GeoDataFrame containing preprocessed roof data with
                     solar potential information and appropriate indexing.
    """
    pv_pot_AEM_geometry = gpd.read_file(join(data_dir, 'solar_potential_AEM.gpkg'))[['DF_UID', 'geometry']]
    pv_pot_AEM = gpd.read_file(join(data_dir, 'gdf.gpkg'))
    pv_pot_AEM = pv_pot_AEM.merge(pv_pot_AEM_geometry, on='DF_UID', suffixes=('_building', ''))
    pv_pot_AEM.dropna(subset=['DF_UID'], inplace=True)
    # set index as 'DF_UID'_'EGID'
    pv_pot_AEM['DF_UID'] = pv_pot_AEM['DF_UID'].astype(int)

    def adjust_values(group):
        # If the group has only one building assigned, return it as is
        if len(group) == 1 or group['DF_UID'].isna().all():
            return group

        # Recalculate the FLAECHE and STROMERTRAG values based on the area proportion
        total_intersec = group['area_intersec'].sum()
        group['FLAECHE'] = group['area_intersec'] / total_intersec * group['FLAECHE'].iloc[0]
        group['STROMERTRAG'] = group['area_intersec'] / total_intersec * group['STROMERTRAG'].iloc[0]
        group['STROMERTRAG_SOMMERHALBJAHR'] = group['area_intersec'] / total_intersec * \
                                              group['STROMERTRAG_SOMMERHALBJAHR'].iloc[0]
        group['STROMERTRAG_WINTERHALBJAHR'] = group['area_intersec'] / total_intersec * \
                                              group['STROMERTRAG_WINTERHALBJAHR'].iloc[0]

        return group

    pv_pot_AEM = pv_pot_AEM.groupby('DF_UID', dropna=False).apply(adjust_values)
    pv_pot_AEM.index = pv_pot_AEM['DF_UID'].astype(str) + '_' + pv_pot_AEM['EGID'].astype(str)
    return pv_pot_AEM


def main():
    """
    Main function that orchestrates the entire PV simulation workflow.

    This function:
    1. Sets up logging
    2. Downloads and extracts building data for Ticino
    3. Processes building registry data
    4. Assigns NAV (Net Annual Value) identifiers to buildings
    5. Filters and categorizes buildings for anonymization
    6. Links PV potential data with building registry data
    7. Runs PV simulations for different scenarios (standard, winter_soft, winter_aggressive, winter_extreme)
    8. Calculates annual and winter energy production metrics
    9. Saves results to various file formats

    The function handles multiple scenarios for PV panel tilt angles to simulate
    different optimization approaches (standard installation, mild winter optimization,
    aggressive winter optimization).
    """
    logger = setup_logger('solar_potential')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    force_profile_calculation = True

    # download the REA data for Ticinella la vita è bella
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
        bldg_raw_gdp.to_file(join(data_dir, 'buildings_raw.gpkg'), driver='GPKG')
        csv_path = join(tmpdirname, 'gebaeude_batiment_edificio.csv')
        bldg_raw = pd.read_csv(csv_path, delimiter='\t', low_memory=False)
    bldg_raw['UNIQUE_PARCEL_ID'] = bldg_raw.apply(lambda x: f"{x['LGBKR']}_{x['LPARZ']}_{x['GGDENR']}", axis=1)
    bldg_built = bldg_raw[bldg_raw['GSTAT'] == 1004]

    egid_nav = pd.read_csv(join(data_dir, 'egid_NAV_raw.csv'), index_col=0)
    for idx, row in tqdm(egid_nav.iterrows(), total=egid_nav.shape[0], desc='Assign NAV parcels'):
        same_egid = bldg_raw.loc[bldg_raw['EGID'] == row['EGID']]
        if same_egid.empty:
            print(f'No parcels found for EGID {row["EGID"]}')
            egid_nav.loc[idx, 'UNIQUE_PARCEL_ID'] = 'unknown'
            continue

        if len(same_egid) > 1:
            print(f'Multiple GSTATs found for EGID {row["EGID"]}')

        # get list of buildings in the same parcel
        egid_nav.loc[idx, 'GSTAT'] = same_egid.iloc[0]['GSTAT']
        egid_nav.loc[idx, 'GGDENR'] = same_egid.iloc[0]['GGDENR']

        if (same_egid['GSTAT'] != 1004).any():
            print(f'EGID {row["EGID"]} is not a built building GSTAT {same_egid["GSTAT"].iloc[0]}')

            # get list of buildings in the same parcel
            built_on_parcel = bldg_built.loc[bldg_built['UNIQUE_PARCEL_ID'] == same_egid.iloc[0]['UNIQUE_PARCEL_ID']]
            if not built_on_parcel.empty:
                egid_nav.loc[idx, 'UNIQUE_PARCEL_ID'] = same_egid.iloc[0]['UNIQUE_PARCEL_ID']
            else:
                egid_nav.loc[idx, 'UNIQUE_PARCEL_ID'] = 'not built'

        else:
            egid_nav.loc[idx, 'UNIQUE_PARCEL_ID'] = same_egid.iloc[0]['UNIQUE_PARCEL_ID']

    # delete buildings that are demolished
    # count demolished buildings
    demolished_buildings = egid_nav[egid_nav['UNIQUE_PARCEL_ID'] == 'not built']
    print(f'Demolished buildings: {len(demolished_buildings)}')
    # unknown parcels
    unknown_parcels = egid_nav[egid_nav['UNIQUE_PARCEL_ID'] == 'unknown']
    print(f'Unknown parcels: {len(unknown_parcels)}')
    egid_nav = egid_nav[(egid_nav['UNIQUE_PARCEL_ID'] != 'not built') & (egid_nav['UNIQUE_PARCEL_ID'] != 'unknown')]

    unique_parcels = egid_nav['UNIQUE_PARCEL_ID'].unique()
    # find egids that are in the unique parcels but are not in the egid_nav
    all_egids = bldg_built[bldg_built['UNIQUE_PARCEL_ID'].isin(unique_parcels)]
    all_egids = all_egids.merge(egid_nav[['EGID', 'Fotovoltaico']].rename(columns={'Fotovoltaico': 'PV'}), on='EGID',
                                how='left')

    # Step 3: Assign NAV_EGID, setting 0 for those missing in egid_nav
    all_egids['NAV_EGID'] = all_egids['EGID']
    all_egids.loc[~all_egids['EGID'].isin(egid_nav['EGID']), 'NAV_EGID'] = 0

    # Step 4: Assign missing NAV_EGIDs based on the same parcel with the biggest 'GAREA'
    # Group by 'UNIQUE_PARCEL_ID' and find the EGID with the largest GAREA in each group
    all_egids['GAREA2'] = all_egids['GAREA'].fillna(0)
    idx_max_area = all_egids.groupby('UNIQUE_PARCEL_ID')['GAREA2'].idxmax()
    all_egids.drop(columns=['GAREA2'], inplace=True)

    # Create a mapping of 'UNIQUE_PARCEL_ID' to the EGID with the largest GAREA
    parcel_to_egid_map = all_egids.loc[idx_max_area, ['UNIQUE_PARCEL_ID', 'EGID']].set_index('UNIQUE_PARCEL_ID')['EGID']

    # Step 5: Fill missing NAV_EGID values with the largest EGID from the same parcel
    all_egids.loc[all_egids['NAV_EGID'] == 0, 'NAV_EGID'] = all_egids['UNIQUE_PARCEL_ID'].map(parcel_to_egid_map)

    # add a columns for buildings with electric heating
    categories_with_heating = [7410, 7411, 7440, 7441, 7450, 7451, 7452]
    all_egids['ELECTRICHEATING'] = all_egids['GWAERZH1'].isin(categories_with_heating)
    # all_egids['EGID_category'] = all_egids.groupby(['GKAT', 'GGDENR', 'GBAUP', 'GAREAC', 'GASTW', 'GANZWHG', 'GWAERZH1', 'GWAERZW1']).ngroup()


    # save the data
    egid_nav.to_csv(join(output_dir, 'egid_NAV.csv'))
    all_egids.to_csv(join(output_dir, 'all_egids.csv'))
    egid_nav.to_pickle(join(output_dir, 'egid_NAV.zip'))
    all_egids.to_pickle(join(output_dir, 'all_egids.zip'))

    bldg_geo_AEM = gpd.GeoDataFrame(all_egids, geometry=gpd.points_from_xy(all_egids.GKODE, all_egids.GKODN))
    # Import data on PV potential (already filtered for Massagno, Isone, Capriasca)
    pv_pot_AEM = load_roofs(data_dir)
    # Assign NAV_EGID
    pv_pot_AEM['NAV_EGID'] = 0
    for i in tqdm(pv_pot_AEM.index, desc='Assigning NAV_EGID'):
        same_egid = bldg_geo_AEM.loc[bldg_geo_AEM['EGID'] == pv_pot_AEM.loc[i, 'EGID'], 'NAV_EGID']
        if len(same_egid) > 0:
            pv_pot_AEM.loc[i, 'NAV_EGID'] = same_egid.iloc[0].astype(int)
    # drop rows with missing NAV_EGID
    pv_pot_AEM = pv_pot_AEM.loc[pv_pot_AEM['NAV_EGID'] != 0]
    return all_egids, egid_nav


if __name__ == "__main__":
    data_dir = 'data/sources'
    output_dir = 'data/processed'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    all_egids, egid_nav = main()
