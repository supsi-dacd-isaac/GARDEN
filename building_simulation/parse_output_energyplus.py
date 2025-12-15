from pathlib import Path
import pandas as pd
import re

def parse_eso(eso_path: Path, input_data: dict) -> tuple[pd.DataFrame, dict]:
    """
    Parse an EnergyPlus ESO file into a timestep DataFrame and a units dict.

    Returns
    -------
    df_timestep : pd.DataFrame
        Timestep-level data with a 'datetime' column added.
    units_dict : dict
        Mapping from variable name -> unit (J converted to kWh).
    """
    with open(eso_path, "r") as f:
        lines = f.readlines()

    # Separate header (data dictionary) and simulation data
    header_lines = []
    sim_lines = []
    header_done = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if "End of Data Dictionary" in line:
            header_done = True
            continue

        if not header_done:
            header_lines.append(line)
        else:
            sim_lines.append(line)

    # Build var_info: frequency, name, unit
    var_info: dict[int, dict] = {}
    for line in header_lines:
        if ("!TimeStep" not in line) and ("!Hourly" not in line):
            continue

        parts = line.split("!")
        main_part = parts[0].strip()
        freq_flag = parts[1]

        if "TimeStep" in freq_flag:
            freq = "timestep"
        elif "Hourly" in freq_flag:
            freq = "hourly"
        else:
            freq = None

        fields = [x.strip() for x in main_part.split(",")]
        try:
            var_num = int(fields[0])
        except ValueError:
            continue

        if len(fields) > 2:
            var_name_full = ", ".join(fields[2:])
        else:
            var_name_full = ""

        unit_match = re.search(r"\[([^\]]+)\]", var_name_full)
        if unit_match:
            unit = unit_match.group(1)
            var_name = re.sub(r"\s*\[[^\]]+\]", "", var_name_full)
        else:
            unit = ""
            var_name = var_name_full

        var_info[var_num] = {"frequency": freq, "name": var_name, "unit": unit}

    # Process simulation data into blocks
    blocks = []
    current_block = None

    for line in sim_lines:
        if not line:
            continue

        # Skip run period info lines
        if line.startswith("1,") and "DEFAULTRUNPERIOD" in line:
            continue

        # A new block starts with record_type "2"
        if line.startswith("2,"):
            if current_block is not None:
                blocks.append(current_block)

            fields = [x.strip() for x in line.split(",")]
            time_info = {
                "record_type": fields[0],
                "day": fields[1],
                "month": fields[2],
                "day_of_month": fields[3],
                "DST": fields[4],
                "hour": fields[5],
                "start_min": fields[6],
                "end_min": fields[7],
                "day_type": fields[8] if len(fields) > 8 else None,
            }
            current_block = {"time": time_info, "data": {}}
        else:
            parts = line.split(",")
            try:
                rec_id = int(parts[0].strip())
            except ValueError:
                continue

            value_str = parts[1].strip()
            try:
                value = float(value_str)
            except Exception:
                value = value_str

            # Convert Joules to kWh
            if (
                rec_id in var_info
                and var_info[rec_id]["unit"] == "J"
                and isinstance(value, float)
            ):
                value = value / 3.6e6

            current_block["data"][rec_id] = value

    if current_block is not None:
        blocks.append(current_block)

    # Build timestep / hourly rows
    timestep_rows = []
    hourly_rows = []

    for block in blocks:
        time_info = block["time"]
        ts_row = time_info.copy()
        hr_row = time_info.copy()

        for rec_id, value in block["data"].items():
            if rec_id not in var_info:
                continue
            freq = var_info[rec_id]["frequency"]
            col_name = var_info[rec_id]["name"]

            if freq == "timestep":
                ts_row[col_name] = value
            elif freq == "hourly":
                hr_row[col_name] = value

        if len(ts_row) > len(time_info):
            timestep_rows.append(ts_row)
        if len(hr_row) > len(time_info):
            hourly_rows.append(hr_row)

    df_timestep = pd.DataFrame(timestep_rows)

    # Build units dict (J already converted to kWh)
    units_dict = {}
    for _, info in var_info.items():
        if info["unit"] == "J":
            info["unit"] = "kWh"
        units_dict[info["name"]] = info["unit"]

    # df_timestep = add_datetime(df_timestep)
    # df_timestep = df_timestep.drop(
    #     columns=[
    #         "record_type",
    #         "day",
    #         "month",
    #         "day_of_month",
    #         "DST",
    #         "hour",
    #         "start_min",
    #         "end_min",
    #         "day_type",
    #     ]
    # )

    return df_timestep, units_dict