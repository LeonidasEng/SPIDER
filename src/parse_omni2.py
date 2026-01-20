import os
from datetime import datetime, timezone, timedelta
import json
from collections import defaultdict
import math

# OMNI2 uses placeholder values to indicate missing or invalid data.
# These must be converted to NaN for numerical analysis later
OMNI_FILL = {99.99, 999.9, 999.99, 9999.0}

# Mapping of OMNI2 physical parameters to column indices
# Note OMNI2 docs use 1-based index, Python is 0-based.
FIELDS = {    
    "bz_gsm": 16,       # IMF Bz (GSM), Word 17 
    "b_mag": 8,         # IMF Magnitude, Word 9
    "v_sw": 24,         # Solar Wind Speed, Word 25
    "np": 23,           # Proton Density, Word 24
    "pdyn": 28,         # Solar Wind Dynamic Pressure, Word 29
    "ey": 35,           # Solar Wind Electric Field, Word 36
    "beta": 36,         # Plasma beta, Word 37
    "mach_alfven": 37,  # Alfven Mach number, Word 38
    "f107": 50          # F10.7 solar radio flux, Word 51
}

def toFloat(val: str):
    '''
    Convert OMNI2 string values to float.
    Replaces known OMNI fill values and invalid entries with NaN.
    '''
    try:
        v = float(val)
        return math.nan if v in OMNI_FILL else v
    except ValueError:
        # Handles missing or malformed fields
        return math.nan

def fileFetch(base_path:str):
    '''
    Generator walks through OMNI2 raw data directory and yields each .dat file
    as a list of lines. Files are sorted chronologically.
    '''
    for root, folders, files in os.walk(base_path):
        files = sorted(files)
        for fname in files:
            if fname.endswith(".dat"):
                path = os.path.join(root, fname)
                with open(path, "r") as f:
                    yield path, f.readlines()

def parseOMNI(text: list):
    '''
    Parse OMNI2 hourly data into a nested dict.
    year -> month -> date (YYYY-MM-DD) -> hour (HH) -> parameters
    '''
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for line in text:
        # Skip emtpy or whitespace-only lines
        if not line.strip():
            continue

        parts = line.split() 
        # OMNI2 record rows must contain at least 51 fields
        if len(parts) < 51:
            continue
        
        # Extract time components
        year = int(parts[0])
        dec_day = int(parts[1]) # Day of year (1-365/366)
        hour = int(parts[2])    # Hour of day (0-23)

        # Convert Decimal day to calendar date
        dt = datetime(year, 1, 1) + timedelta(days=dec_day - 1)
        date_key = dt.strftime("%Y-%m-%d")
        month = dt.month
        hour_key = f"{hour:02d}"

        # Extract selected physical parameters
        record = {}
        for name, idx in FIELDS.items():
            record[name] = toFloat(parts[idx])
        
        # Store hourly record
        out[year][month][date_key][hour_key] = record
    
    return out

def dumpJob(year: int, month: int, month_data: dict, out_base: str):
    '''
    Write a single monthly OMNI2 JSON file.
    '''
    out_dir = os.path.join(out_base, str(year))
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, f"omni2_{year}_{month:02d}.json")

    # Sort by date (YYYY-MM-DD)
    sorted_days = dict(sorted(month_data.items()))

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sorted_days, f, indent=4)

    print(f"Dumped OMNI2 {year}-{month:02d} -> {out_file}")

def main():
    base = os.environ.get("SPIDER")
    if base is None:
        raise EnvironmentError("SPIDER system variable is not set")
    
    raw_path = os.path.join(base, "data/raw/nasa_omni/omni2/20220101_20251231_raw")
    out_path = os.path.join(base, "data/data_processed/omni2")

    # Aggregated structure: year -> month -> date -> hour
    yearly_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for _, text in fileFetch(raw_path):
        parsed = parseOMNI(text)
        for year, months in parsed.items():
            for month, days in months.items():
                yearly_data[year][month].update(days)
    
    # Write one JSON file per month
    for year, months in yearly_data.items():
        for month, month_data in months.items():
            dumpJob(year, month, month_data, out_path)

if __name__ == "__main__":
    main()