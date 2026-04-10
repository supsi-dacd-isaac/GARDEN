# app/energyplus_runner.py
import subprocess
from pathlib import Path
# from .config import ENERGYPLUS_CMD, WEATHER_FILE, OUTPUT_DIR
import pprint
import os

BASE_DIR = Path(__file__).resolve().parent
WEATHER_FILE = BASE_DIR / "weather" / "Tesserete_2020.epw"
OUTPUT_DIR = BASE_DIR / "output"

# EnergyPlus CLI command - set via ENERGYPLUS_CMD environment variable
ENERGYPLUS_CMD = os.getenv("ENERGYPLUS_CMD")
if not ENERGYPLUS_CMD:
    raise EnvironmentError("ENERGYPLUS_CMD environment variable not set. Set it to your EnergyPlus executable path.")

def run_energyplus(idf_path: Path) -> Path:
    """
    Calls:
      energyplus -w /weather/site.epw -d /output generated.idf
    Returns path to .eso file.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    cmd = [
        ENERGYPLUS_CMD,
        "-w", str(WEATHER_FILE),
        "-d", str(OUTPUT_DIR),
        "-p", idf_path.stem, # <– use the IDF filename as prefix
        "-j", str(2),
        str(idf_path)
    ]
    subprocess.run(cmd, check=True)
    # EnergyPlus names the .eso same as the IDF basename:
    #pprint.pprint(idf_path.stem) # deb
    eso_file = OUTPUT_DIR / f"{idf_path.stem}out.eso"
    #csv_file = OUTPUT_DIR / f"{idf_path.stem}tbl.csv"
    #pprint.pprint(eso_file) # deb
    if not eso_file.exists():
        raise FileNotFoundError(f"Expected .eso at {eso_file}")
    #return eso_file, csv_file    
    return eso_file