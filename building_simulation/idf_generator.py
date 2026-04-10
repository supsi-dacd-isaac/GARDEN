# to be fixed:
# - when there's more than 1 floor, the ceiling between floors should be properly defined as ceiling, and not as 'roof'
# - Surrounding buildings should be rendered

from math import sqrt
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
# from .config import TEMPLATES_DIR
import numpy as np
# from .adjacency_calculator import calculate_adjacency_matrix_from_lat_lon

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
PROFILES_DIR = TEMPLATES_DIR / "profiles"

env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), 
                  trim_blocks=True, lstrip_blocks=True)


def assign_infiltration_rate(str_year_construction):

    # ref is CESARP database: https://github.com/hues-platform/cesar-p-core/blob/master/src/cesarp/graphdb_access/ressources/construction_and_material_data.ttl/
    adict_infiltration_rate = {'1918':0.55,
                               '1918-1948':0.71,
                               '1949-1980':0.76,
                               '1981-1994':0.59,
                               '1995-2000':0.39,
                               '2001-2010':0.28,
                               '2011':0.16,
                               '>2011':0.16}
    
    afloat_infiltration_rate = adict_infiltration_rate[str_year_construction]

    return afloat_infiltration_rate


def assign_flow_temperature(str_year_construction, emitter):

    
    if emitter == 'radiantFloor':
        # ref is https://pubdb.bfe.admin.ch/en/publication/download/9982
        adict_flow_temp = {'1918':42.5,
                                '1918-1948':42.5,
                                '1949-1980':42.5,
                                '1981-1994':42.5,
                                '1995-2000':35,
                                '2001-2010':35,
                                '2011':32.5,
                                '>2011':32.5}
    

    else: # radiator

        adict_flow_temp = {'1918':65,
                                '1918-1948':65,
                                '1949-1980':65,
                                '1981-1994':55,
                                '1995-2000':55,
                                '2001-2010':45,
                                '2011':37.5,
                                '>2011':37.5}
    
    afloat_flow_temperature = adict_flow_temp[str_year_construction]

    return afloat_flow_temperature


def assign_residents_per_dwelling(building_floor_area, dwelling_number, total_floors):

    # ref is https://www.bfs.admin.ch/bfs/it/home/statistiche/costruzioni-abitazioni/abitazioni/condizioni-abitazione/densita-utilizzazione.assetdetail.36158269.html
    
    avg_area_per_dwelling = round((building_floor_area * total_floors)/dwelling_number,1)

    if avg_area_per_dwelling < 30:
        return 1.3
    elif 30 <= avg_area_per_dwelling < 40:
        return 1.4
    elif 40 <= avg_area_per_dwelling < 60:
        return 1.5
    elif 60 <= avg_area_per_dwelling < 80:
        return 1.7
    elif 80 <= avg_area_per_dwelling < 100:
        return 2.0
    elif 100 <= avg_area_per_dwelling < 120:
        return 2.2
    elif 120 <= avg_area_per_dwelling < 160:
        return 2.4
    else:
        return 2.7
    




def generate_idf(input_data: dict) -> Path:
    """
    - Loads base.idf.j2
    - Loads Construction component snippet .j2 based on input_data['construction_period'] (1918, 1945, 1978, etc.)
    - Loads HVAC component snippet .j2 based on input_data['heatingSystem','heatingEmitter','storage','solarCollector']
    - Loads other component snippet .j2 (lights, people, infiltration, etc.)
    - Renders and injects all snippets into base at the different markers {{ hvac }}, {{ construction }}, etc.
    - Writes to output/generated.idf
    
    """

    # TOTAL_FLOORS_LEN  : N number of total floors of the building
    # TOTAL_FLOORS_LIST : [0,1,...,N-1] list of N-1 floors 
    # IS_HEATPUMP_SH       : bool to configure HVAC if True HP if False boiler
    # IS_RADIANTFLOOR   : bool to configure HVAC if True radiant floor if False wall radiators
    # IS_SOLAR          : bool to configure HVAC if True solar collectors for dhw
    # IS_TANK_DHW       : bool to configure HVAC if False no dhw tank and hot water comes directly from the boiler/HP
    # IS_TANK_SH        : bool to configure HVAC if False no sh tank and hot water comes directly from the boiler/HP
    

    # copy the json sent by the frontend
    input_data = input_data.copy()

    # Pass the profiles directory path for Schedule:File references (cross-platform)
    input_data['profiles_dir'] = str(PROFILES_DIR)

    emitter = input_data['emitter']

    # If it's not calculated from the location+REA+script, get default values north + all walls exposed:
    orientation = input_data.get('buildingOrientation', 'north') # deviation from N orientation in deg (clockwise)
    exposedWalls = input_data.get('buildingMorphology', '4')


    TOTAL_FLOORS_LEN    = input_data['totalFloors']
    TOTAL_FLOORS_LIST   = list(range(0,TOTAL_FLOORS_LEN))
    IS_HEATPUMP_SH      = True if input_data['shSystem'] == 'hp' else False
    IS_HEATPUMP_DHW     = True if input_data['dhwSystem'] == 'hp' else False
    IS_RADIANTFLOOR     = True if emitter == 'radiantFloor' else False
    IS_SOLAR            = True if input_data['dhwSolarThermal'] and input_data['dhwSolarThermalArea'] > 0 else False
    IS_TANK_DHW         = True if input_data['dhwVolume'] > 0 else False
    IS_TANK_SH          = True if input_data['shVolume'] > 0 else False
    IS_SINGLEFAMILY     = True if input_data['buildingType'] == 'single-family' else False
    IS_DWELLING_RESIDENTS = False if input_data['dwellingResidents'] == '' else True

    input_data['TOTAL_FLOORS_LEN']  = TOTAL_FLOORS_LEN
    input_data['TOTAL_FLOORS_LIST'] = TOTAL_FLOORS_LIST
    input_data['IS_HEATPUMP_SH']    = IS_HEATPUMP_SH
    input_data['IS_HEATPUMP_DHW']   = IS_HEATPUMP_DHW
    input_data['IS_RADIANTFLOOR']   = IS_RADIANTFLOOR
    input_data['IS_SOLAR']          = IS_SOLAR
    input_data['IS_TANK_DHW']       = IS_TANK_DHW
    input_data['IS_TANK_SH']        = IS_TANK_SH
    input_data['IS_SINGLEFAMILY']   = IS_SINGLEFAMILY
    input_data['IS_DWELLING_RESIDENTS'] = IS_DWELLING_RESIDENTS

    # Energyplus simulation parameters (default)
    input_data['timestep'] = 4
    input_data['minIte'] = 2 # def. 2
    input_data['maxIte'] = 8 # def. 8


    # Define some model standard variables
    input_data['capacitanceMultiplier'] = 8
    input_data["windowRatio"] = 0.16
    input_data["windowHeightRatio"] = 0.5 # for the windows
    input_data["floorHeight"] = 2.6
    input_data['infiltrationRate'] = assign_infiltration_rate(input_data['constructionPeriod']) # Air Changes per Hour {1/hr}
    input_data['ventilationRate'] = 0.0009144 # Flow per square meters {m3/(s·m2)} !!!!! REVIEW: this is not ventilation, but fresh air needs
    input_data['heating_sizing_factor'] = 1.5 # Coefficient to oversize the whole system with respect to the design days
    coef_heating = 0.85 # missing ref. (floor surface heated ratio)
    input_data['delT'] = 0.25 # C IT CAN PROBABLY ME REMOVED (it belongs to V1)
    input_data['floor_opening_ratio'] = 0.1 # Fraction of the floor area that is open (e.g., stairwell or double-height void) connecting adjacent floors in single-family houses FIND REF
    

    # calculate adjacency and REA area of all dwellings
    # adj_gdf = calculate_adjacency_matrix_from_lat_lon(input_data['lon'],input_data['lat'], input_data['egid'])
    # adj_matrix = adj_gdf[adj_gdf['egid'] == str(input_data['egid'])]
    # input_data['buildingArea'] = adj_matrix['area_dwellings'].iloc[0]
    # input_data['adj_matrix'] = adj_matrix['adj_matrix'].iloc[0]
    #
    # print(input_data['adj_matrix'][0][0])
    #print(input_data['buildingArea'])



    # Calculate number of residents per building
    avg_residents_per_dwelling = assign_residents_per_dwelling(input_data['buildingFloorArea'], input_data['dwellingNumber'], TOTAL_FLOORS_LEN)
    if IS_DWELLING_RESIDENTS: # if the field 'dwellingResidents' has been defined by the user
        input_data['buildingResidents'] = input_data['dwellingResidents'] if IS_SINGLEFAMILY else (input_data['dwellingResidents'] + (input_data['dwellingNumber']-1)*avg_residents_per_dwelling) #2p per household in avg.
    else: # if the user has not defined it manually (e.g. when running regional analyses)
        input_data['buildingResidents'] = input_data['dwellingNumber'] * avg_residents_per_dwelling
    input_data['zoneResidents'] = input_data['buildingResidents']/TOTAL_FLOORS_LEN

    #print(avg_residents_per_dwelling)
    #print(input_data['buildingResidents'])

    # Define the way to control Temperatures (air temperature/air+radiation temperature)
    #thermostat_method = 'Air'
    thermostat_method = 'Operative'
    input_data['THERMOSTAT_METHOD'] = thermostat_method

    # Calculate some model parameters
    if IS_SINGLEFAMILY:
        input_data['buildingHeatingArea'] = input_data['dwellingArea'] * coef_heating if input_data['dwellingArea'] > 0 else input_data['buildingArea'] * coef_heating
        input_data['floorHeatingArea'] = input_data['buildingHeatingArea'] / input_data['totalFloors']
    else:
        if input_data['buildingArea'] > 0: # if the data comes from REA dwellings
            input_data['buildingHeatingArea'] = input_data['buildingArea'] * coef_heating
            input_data['floorHeatingArea'] = input_data['buildingHeatingArea'] / input_data['totalFloors']
        else:
            input_data['floorHeatingArea'] = input_data['buildingFloorArea'] * coef_heating # heating area per floor. Value used to define the geometry of the building.
            input_data['buildingHeatingArea'] = input_data['floorHeatingArea'] * input_data['totalFloors']


    # calculate building orientation

    if orientation == 'north':
       northAxis = 0
    elif orientation == 'east':
       northAxis = 90
    elif orientation == 'south':
       northAxis = 180
    else:  # west
       northAxis = 270
    input_data['northAxis'] = northAxis
    # input_data['northAxis'] = 0

    # Calculate building insulation 
    if exposedWalls == '4':
       adiabaticWalls = 0
    elif exposedWalls == '3':
       adiabaticWalls = 1
    elif exposedWalls == '2':
       adiabaticWalls = 2
    elif exposedWalls == '1':
       adiabaticWalls = 3
    input_data['ADIABATICWALLS'] = adiabaticWalls
    # input_data['ADIABATICWALLS'] = 0


    # Add stairwell or double-height void connecting adjacent floors in single-family houses
    if IS_SINGLEFAMILY:
        solid_floor_fraction = 1-input_data['floor_opening_ratio']
        solid_side_ratio = sqrt(solid_floor_fraction) # because we modify the "side" of the geometry
    else:
        solid_side_ratio = 1
    input_data['solid_side_ratio'] = solid_side_ratio


    # Define HP model
    hp_list = ['aerotop_g07_14m','aerotop_t35r']
    if IS_SINGLEFAMILY:
        target_hp_model = 0 # 
    else:
        target_hp_model = 1
    input_data['hp_model'] = hp_list[target_hp_model]

    IS_AEROTOP_G07_14M = True if input_data['hp_model'] == 'aerotop_g07_14m' else False
    if IS_AEROTOP_G07_14M:
        input_data['IS_AEROTOP_G07_14M']   = IS_AEROTOP_G07_14M
        input_data['hp_ref_capacity'] = 14340
        input_data['hp_ref_cop'] = 6.49
    
    IS_AEROTOP_T35R = True if input_data['hp_model'] == 'aerotop_t35r' else False
    if IS_AEROTOP_T35R:
        input_data['IS_AEROTOP_T35R']   = IS_AEROTOP_T35R
        input_data['hp_ref_capacity'] = 39600
        input_data['hp_ref_cop'] = 4.1

    print('HP model: ', input_data['hp_model'])


    # Define water temperatures.
    # TO DO: temperature of the flow depending on year of construction

    # IF THE DHWT T SETPOINT IS <= 55, THE HP WORKS ALONE, IF IS HIGHER, THE AUX HEATER WILL WORK. 
    dhw_setpoint = input_data['dhwSetpoint'] # IF IT'S UNDER 55C THE AUX HEATER IS NOT ACTIVATED
    hp_setpoint_limit = 55
    input_data['hp_setpoint_limit'] = hp_setpoint_limit
    input_data['dhw_flow_temperature'] = dhw_setpoint # (def. 60) 
    input_data['dhw_tank_setpoint'] = dhw_setpoint # (def. 60)
    input_data['dhw_use_temperature'] = dhw_setpoint # (def. 60)
    input_data['dhw_target_temperature'] = dhw_setpoint # (def. 60)
    
    sh_water_temperature = assign_flow_temperature(input_data['constructionPeriod'],emitter)
    #print(sh_water_temperature)
    # Rated parameters for radiators
    input_data['ratedWaterTemperature'] = sh_water_temperature 
    input_data['ratedWaterMassFlowRate'] = 0.063 # 0.063 default
    # Rated params for all systems  
    input_data['sh_flow_temperature'] = sh_water_temperature
    input_data['sh_tank_setpoint'] = sh_water_temperature


    # Define the period to increase DHW and SH temperatures if flexibility
    #nput_data["increaseTankTemperature"] = 5
    input_data['hour_start_increase'] = '12:00'
    input_data['hour_stop_increase'] = '17:00'


    # Define efficiency of boilers
    # TODO: should be adapted to fuel, if condensation, and year of installation.
    input_data['boilerEfficiency'] = 0.88
    

    # Fix sh and dhw fuels in case are not 'Electricity' with HP:
    if IS_HEATPUMP_SH:
        input_data['shFuel'] = 'Electricity'
    if IS_HEATPUMP_DHW:
        input_data['dhwFuel'] = 'Electricity'


    # Define auxiliary DHW heater power
    if IS_HEATPUMP_DHW: 
        # boiler works as aux
        input_data['auxHeaterCapacity'] = 500*input_data['buildingResidents'] # (500-1000)
    else: 
        # boiler works as main heater
        input_data['auxHeaterCapacity'] = 1500*input_data['buildingResidents']
       
    #input_data['auxHeaterCapacity']=0.1
    
    
    # Define other sensible parameters
    # Time (hr) required by the tank to be refilled: BE CAUTIOUS WITH THIS 
    input_data['tank_dhw_recovery_time'] = 1.5 # Indirect Water Tank Recovery Time {hr}
    input_data['tank_sh_recovery_time'] = 0.1 # Indirect Water Tank Recovery Time {hr}
    # EMS Tank Hysteresis (degC) to switch between DHW and SH
    input_data['hyster_dhw_flex'] = 5 # flex 4-7 is ok. SYSTEM SENSITIVE
    input_data['hyster_dhw'] = 5 # no flex 4-5 is ok. SYSTEM SENSITIVE
    input_data['hyster_sh'] = 1  # 0.5-1.5 is ok SYSTEM SENSITIVE
    
    
    # Define appliances and light consumption per square meter (or change to consumption / number of residents)
    input_data['appliances_power_square_meter'] = 8
    input_data['light_power_square_meter'] = 2.74


    

    # calculate window dimensions and location in each wall
    side   = round(sqrt(input_data["floorHeatingArea"]),2) # wall length (assuming building is a square)
    input_data['side'] = side
    height = input_data["floorHeight"]
    wall_area = round(side * height,2)
    win_area  = round(wall_area * input_data["windowRatio"],2) # wall/window ratio in m/m 
    H_win     = round(height * input_data["windowHeightRatio"],2) # window height/width ratio m/m
    W_win     = round(win_area / H_win,2)
    input_data["W_win"]   = round(W_win,2)
    input_data["H_win"]   = round(H_win,2)
    input_data["x_off"]   = round((side - W_win) / 2,2)
    input_data["z_off"]   = round((height - H_win) / 2,2)


    # Calculate solar panel geometry
    panel_tilt = 30 # degrees
    panel_length = 2.2
    panel_width = input_data['dhwSolarThermalArea']/panel_length if input_data['dhwSolarThermalArea'] > 0 else 0.1
    input_data['height_panel'] = round(panel_length * np.sin(np.pi/2*panel_tilt/90),2)
    input_data['width_panel'] = panel_width
    input_data['length_panel'] = panel_length    
    # Define solar panel water temperature
    input_data['solar_flow_temperature'] = 90


    # Define Flexibility parameters (THIS PART IS USED IF FLEX SCENARIO)
    setpoint = input_data['shSetpoint']
    global_reduction_temp = input_data['reduceGlobalSetpoint']
    peak_reduction_temp = input_data['reducePeakSetpoint']

    #input_data["DRstart"] = "17:00" # this is now defined in the input JSON
    #input_data["DRend"] = "22:00" # this is now defined in the input JSON
    
    # Calculate Global Temperature Adjustment
    setpoint = setpoint - global_reduction_temp
    input_data['shSetpoint'] = setpoint
    
    # Calculate Peak reduction    
    drStartHour = int(input_data['DRstart'].split(":", 1)[0])
    #drEndHour = int(input_data['DRend'])
    #input_data['DRstart'] = f'{drStartHour}:00'
    #input_data['DRend'] = f'{drEndHour}:00'
    input_data['DRsetpoint'] = setpoint - peak_reduction_temp # setpoint variation
    # Preheating
    if input_data['increasePreheatingSetpoint'] != 0:
        prehLength = 5 #hours
        prehStart = drStartHour - prehLength
        prehFinal = drStartHour
        timesteps = list(range(prehStart*60, prehFinal * 60, 15))
        prehT = setpoint + input_data['increasePreheatingSetpoint']
        Tdelta = input_data['increasePreheatingSetpoint'] / (len(timesteps)-1)
        Tempsteps = np.arange(setpoint, prehT+Tdelta, Tdelta)
        list_hours = []
        list_temps = []
        list_temps_low = []
        for i, mins in enumerate(timesteps):
            h, m = divmod(mins, 60)
            list_hours.append(f"{h:02d}:{m:02d}")
            list_temps.append(str(float(round(Tempsteps[i],2))))
            list_temps_low.append(str(float(round(Tempsteps[i]-input_data['delT'],2))))
        input_data['list_hours'] = list_hours
        input_data['list_temps'] = list_temps
        input_data['list_temps_low'] = list_temps_low
    else:
        input_data['list_hours'] = []
        input_data['list_temps'] = []
        input_data['list_temps_low'] = []
    #print(list_hours)


    # Calculate DHW demand
    dhw_person_day = 50 # liters per person per day
    input_data['dhw_person_day'] = dhw_person_day
    dhw_zone_day = dhw_person_day * input_data['buildingResidents'] / TOTAL_FLOORS_LEN # Even though we have 3 zones, the consumption is the one for the whole building divided by the total zones/floors   
    dhw_m3_s_zone = dhw_zone_day / 1000 / 3600 # at this rate, during 1h, you get the total dhw consumption of each zone/floor
    input_data['dhw_rate_zone'] = dhw_m3_s_zone # m3/s hot water per zone
    

    # Defining templates

    # Defining constuction according to period of construction and renovation/no renovation 
    #construction_period_template = f"construction_{input_data['constructionPeriod'].replace('-','_')}" # "1995-2001"
    #construction_period_template = f"construction_{input_data['constructionPeriod'].replace('<','_')}" # "<1918"
    construction_period_template = []
    for element in ['walls','floors','roof','windows']:
        if input_data[f'{element}RenovationPeriod'] == 'noRenovation':
            period_element = input_data['constructionPeriod']
        else:
            period_element = input_data[f'{element}RenovationPeriod']
        if (element == 'floors') & (input_data['emitter'] == 'radiantFloor'):
            element = element+'_radiant'
        construction_template = f"construction_{element}_{period_element.replace('-','_').replace('<','_')}"
        #print(construction_template)
        construction_period_template.append(construction_template)


    # Defining lighting template
    lights_template = "lights"

    # Defining People template
    people_template = "people"

    # Defining Appliances template
    appliances_template = "appliances"

    # Defining infiltration template
    infiltration_template = "infiltration"

    # Defining ventilation template
    ventilation_template = "ventilation"

    # Defining dhw template
    #dhw_template = "dhw"

    # Defining heating system template - OLD VERSION
    # Right now, we define the same Heating Sys for each household but this can be modified in the future
    #heating_system = input_data["heatingSystem"]
    #emitter = input_data["heatingEmitter"]
    #storage = input_data["heatingStorage"]
    #solar = input_data['solarThermal']

    #if storage == True:
    #    if solar:
    #        hvac_template = f"hvac_{heating_system}_{emitter}_storage_solar"
    #    else:
    #        hvac_template = f"hvac_{heating_system}_{emitter}_storage"
    #else:
    #    if solar:
    #        hvac_template = f"hvac_{heating_system}_{emitter}_solar"
    #    else:
    #        hvac_template = f"hvac_{heating_system}_{emitter}"
    
    # Defining heating system common instances template (to avoid generating the same instances for each floor)
    #hvac_common_instances_template = "hvac_common_instances"

    # HVAC template
    hvac_template = "hvac"

    # Outputs template
    outputs_template = "outputs"

    # Schedule limits (E+ requirement)
    schedule_type_limits = "scheduleTypeLimits"

    # hp_curves
    hp_curves = "hp_curves"

    # hp_curves
    boiler_curves = "boiler_curves"

    # EMS
    ems = "ems"

    # thermostat
    thermostat = "thermostat"

    # pre-render each template
    snippets = {}
    snippets_list = [
        construction_period_template[0], # construction element Wall
        construction_period_template[1], # construction element Floor
        construction_period_template[2], # construction element Roof
        construction_period_template[3], # construction element Window
        schedule_type_limits,
        hvac_template, # hvac specific for the heating system
        hp_curves,
        boiler_curves,
        ems,
        thermostat,
        #hvac_common_instances_template, # hvac common in all heting systems
        lights_template, # light use 
        people_template, # people occupancy
        appliances_template, # appliances use
        infiltration_template, # infiltration
        ventilation_template,  # ventilation
        #dhw_template, # dhw
        outputs_template # outputs
        ]
    
    for piece in snippets_list:
        print(piece)
        tpl = env.get_template(f"{piece}.idf.j2")
        #print(tpl)
        snippets[piece] = tpl.render(**input_data)

    # render base 
    base_tpl = env.get_template(f"base.idf.j2")
    
    
    # full text
    full_txt = base_tpl.render(**input_data, **snippets)


    #full_txt = base_txt #debugging

    out_path = Path("output") / "generated.idf"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(full_txt)

    return out_path, input_data
    