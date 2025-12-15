import random
import json
import os

def generate_template():
    building_types = ["multi-family", "single-family"]
    building_type = random.choice(building_types)
    morphologies = ["1", "2", "3", "4"]
    orientations = ["north", "south", "east", "west"]
    construction_periods = ["1918", "1918-1948", "1949-1980", "1981-1994", "1995-2000", "2001-2010", "2011"]
    renovation_periods = ["noRenovation"] + construction_periods
    heating_systems = ["boiler", "hp"]
    heating_fuels = ["Diesel", "NaturalGas", "Electricity"]
    heating_emitters = ["radiantFloor", "radiators"]
    dhw_systems = ["boiler", "hp"]
    at_home_options = ["rarely", "often"]

    total_floors =  random.randint(1, 3 if building_type == "single-family" else 5)
    building_floor_area = random.randint(30, 100)

    template = {
        "buildingType": building_type,
        "totalFloors": total_floors,
        "buildingMorphology": random.choice(morphologies),
        "buildingOrientation": random.choice(orientations),
        "constructionPeriod": random.choice(construction_periods),
        "wallsRenovationPeriod": random.choice(renovation_periods),
        "floorsRenovationPeriod": random.choice(renovation_periods),
        "roofRenovationPeriod": random.choice(renovation_periods),
        "windowsRenovationPeriod": random.choice(renovation_periods),
        "buildingFloorArea": building_floor_area,
        "dwellingNumber": 1 if building_type == "single-family" else 2 * total_floors,
        "dwellingArea": 100, #FIXME
        "dwellingResidents": random.randint(1, 5),
        "shSetpoint": random.randint(18, 24),
        "shSystem": random.choice(heating_systems),
        "shFuel": random.choice(heating_fuels),
        "emitter": random.choice(heating_emitters),
        "shVolume": random.choice([0, random.randint(50, 500)]), #FIXME
        #"solarThermal": random.choice([True, False]),
        # "solarCollectorArea": random.randint(0, 20),
        # "solarTankVolume": random.randint(0, 500),
        "dhwSystem": random.choice(dhw_systems), #FIXME
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
        "flexibleHeating": False #FIXME

    }

    return template


path = "automatically_generated"
# Create output directory if not exists
os.makedirs(path, exist_ok=True)

# Generate and save 25 templates
for i in range(1, 3):
    template = generate_template()
    filename = f"{path}/test_{i}.json"
    with open(filename, "w") as f:
        json.dump(template, f, indent=2)
    print(f"✅ Saved {filename}")
