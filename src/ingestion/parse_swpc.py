import os
import json
from datetime import datetime, timedelta
from collections import defaultdict


# The kind people at NOAA provided me with this data to fill in the gaps that I found within the NCEI archive
# using an in-house tool. This data is not definitive and may contain errors.

DATA = {
    "Full":"full_data_2011-2015.json"
    # Yes only one file now, but if this should change when integrating test data
    # additional entries will be easy to attach
}

def importFile(file_path:str):
    """
    Imports and parses a JSON file containing SWPC forecast data.

    Args: 
        file_path (str): Path to the JSON file to import.

    Returns:
        dict | list: Parsed JSON data loaded from the file.
    
    Raises:
        ValueError: If the JSON file cannot be found at the specified path.
    """
    try:
        # Helper function for reading in JSON file
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        # Should not occur as only 3 files, but good practice
        raise ValueError(f"JSON file not found at: {file_path}.")
    
def classifyStorm(kp:float):
    """
    Converts Kp value into the corresponding NOAA geomagnetic storm
    scale classification.

    Args:
        kp (float): Kp value
    
    Returns:
        str | None
            NOAA geomagnetic storm scale:
            - `G1` to `G5`
            - Returns `None` if the Kp value is below storm threshold
            (`Kp < 5`). 
    """
    if kp >= 9: return "G5"
    if kp >= 8: return "G4"
    if kp >= 7: return "G3"
    if kp >= 6: return "G2"
    if kp >= 5: return "G1"
    return None

def normaliseRecord(data: dict) -> dict:
    """
    Normalises a SWPC forecast record into the SPIDER forecast schema.

    Args:
        data (dict): Raw SWPC forecast record.
    
    Returns:
        dict:
            Normalised forecast record containing:
            - `issue_time_utc`
            - `valid_start_utc`
            - `lead_time_hrs`
            - `forecast_kp`
    Notes:
        Observed Kp values are omitted because they available using
        another script: `parse_dayind.py`
    """
    # Normalise each record using the known SPIDER schema
    issue = datetime.fromisoformat(data["cycle"].replace("Z", "+00:00"))
    valid = datetime.fromisoformat(data["valid"].replace("Z", "+00:00"))

    return {
        "issue_time_utc": str(issue),
        "valid_start_utc": str(valid),
        "lead_time_hrs": data["tau"],
        "forecast_kp": data["forecastKp"]
        # Omitting observed as not present in other 3day forecast products
    }

def splitJSON(threeday_json:list):
    """
    Splits SWPC forecast records into `0030` and `1230` issue groups.

    Args:
        threeday_json (list): List of raw SWPC forecast records.
    
    Returns:
        tuple[list, list]
            - threeday_0030: Forecast records issued before 12:00 UTC.
            - threeday_1230: Forecast records issued at or after 12:00 UTC.
    
    Raises:
        ValueError: if an unexpected issue time error occurs.
    """
    # Keep as lists to allow sorting later
    threeday_0030 = []
    threeday_1230 = []

    for data in threeday_json:
        record = normaliseRecord(data)

        issue_date, issue_time = record["issue_time_utc"].split()
        hour = int(issue_time.split(":")[0])
        # Extract hour from time (some issues are late but this is exception not rule)
        if  hour < 12:
            threeday_0030.append(record)
        elif hour >= 12:
            threeday_1230.append(record)
        else:
            # There should be only two issues
            raise ValueError("Unexpected issue time detected.")

    return threeday_0030, threeday_1230

def sortJSON(threeday_0030, threeday_1230):
    """
    Sorts forecast records by issue time and valid start time.

    Args:
        threeday_0030 (list): Forecast records associated with `0030` issue period.
        threeday_1230 (list): Forecast records assocated with `1230` issue period.
    
    Returns:
        tuple[list, list]
            Chronologically sorted forecast records for each issue period.
    """
    # Key helper function making sorting neater
    def sorter(record):
        return (record["issue_time_utc"], record["valid_start_utc"])
    
    sorted_0030 = sorted(threeday_0030, key=sorter) # Each record passed uses helper tuple
    sorted_1230 = sorted(threeday_1230, key=sorter)

    return sorted_0030, sorted_1230

def buildForecastStruct(sorted_records:list):
    """
    Builds a structured forecast dictionary compatible with the existing SPIDER
    3-day forecast processing pipeline.

    Args:
        sorted_records (list): Chronologically sorted SWPC forecast records.

    Returns:
        dict:
            Structured forecast dictionary grouped by issue date and forecast lead
            period.
    
    Notes:
        Forecasts are grouped into: `n`, `n+1`, `n+2` based on the different lead times
        from issue date (`n`).

        Forecast entries are stored as:

        - forecast time bin
        - forecast Kp value
        - NOAA G-scale classification.

        Solar radiation, radio blackout and metadata sections are omitted
        because they were not included in the supplied SWPC dataset.
    """
    # Initialise the default structure shared by processed data
    # Omitting meta, Solar and Radio data as this was not provided
    output = defaultdict(lambda: {
        "issue": None,
        "kp": {"n": [], "n+1": [], "n+2": []}
    })

    for record in sorted_records:
        issue = datetime.fromisoformat(record["issue_time_utc"])
        valid = datetime.fromisoformat(record["valid_start_utc"])

        dt_key = issue.strftime("%Y-%m-%d") # Key
        issue_str = issue.strftime("%Y-%m-%d %H:%M:%S UTC") # meta

        issue_date = issue.date()
        valid_date = valid.date()

        lead_day = (valid_date - issue_date).days

        if lead_day not in (0, 1, 2):
            continue

        forecast_bin = ["n", "n+1", "n+2"][lead_day]

        end = valid + timedelta(hours=3)
        label = f"{valid.strftime('%Y-%m-%d %H')}-{end.strftime('%H')}UT"

        kp = float(record["forecast_kp"])
        g = classifyStorm(kp)

        output[dt_key]["issue"] = issue_str
        output[dt_key]["kp"][forecast_bin].append([label, kp, g])
    
    return dict(output)

def dumpJob(sorted_records:list, proc_output:str, tag:str):
    """
    Writes processed SWPC forecast data into monthly JSON files.

    Args:
        sorted_records (list): Sorted forecast records to export.
        proc_output (str): Root processed data output directory.
        tag (str): Forecast issue period identifier (`0030` or `1230`).
    
    Returns:
        None
    
    Notes:
        Output files are written using the structure:
        `operation_full/3day_<tag>/<year>/3day_<year>_<month>.json`
    """
    out_dir = os.path.join(proc_output, "operation_full", f"3day_{tag}")

    struct = buildForecastStruct(sorted_records)
    
    monthly_files = {}

    for day, content in struct.items():
        year = day[:4] # Year substring
        month = day[5:7] # Month substring

        year_dir = os.path.join(out_dir, year)
        os.makedirs(year_dir, exist_ok=True)

        file_path = os.path.join(year_dir, f"3day_{year}_{month}.json")

        # If path does not exist create new one
        if file_path not in monthly_files:
            monthly_files[file_path] = {}
        
        # Populate monthly dict
        monthly_files[file_path][day] = content

    for file_path, data in monthly_files.items():
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dict(sorted(data.items())), f, indent=4)
        
        print(f"Dumped {tag} data to {file_path}")

def main():
    """
    Main entry point for processing supplied SWPC historical forecast
    datasets.

    The pipeline:
    - Loads supplied SWPC JSON forecast datasets.
    - Splits forecasts into `0030` and `1230` issue periods.
    - Sorts forecasts chronologically.
    - Converts records into the SPIDER forecast schema.
    - Writes processed monthly forecast JSON files.

    Notes:
        Raw SWPC forecast datasets are read from:
        `data/raw/forecasts/3day/operation_full`

        This data was provided through direct communication with NOAA and generated using 
        an internal tool representing the available historical Kp forecast archive.

        The dataset is not considered definitive and may contain errors.
    
    Raises:
        EnvironmentError: If the `SPIDER` environment variable is not defined.
    """
    base = os.environ.get("SPIDER")
    if base is None:
        raise EnvironmentError("SPIDER system variable is not set!")
    
    raw_path = os.path.join(base, "data", "raw", "forecasts", "3day", "operation_full")
    processed_path = os.path.join(base, "data", "data_processed", "3day_forecast")

    for _, file_name in DATA.items():
        threeday_json = importFile(os.path.join(raw_path, file_name))
        threeday_0030, threeday_1230 = splitJSON(threeday_json)
        sorted_0030, sorted_1230 = sortJSON(threeday_0030, threeday_1230)
        # Call dump job for each issue type
        dumpJob(sorted_0030, processed_path, "0030")
        dumpJob(sorted_1230, processed_path, "1230")

if __name__ == "__main__":
    main()