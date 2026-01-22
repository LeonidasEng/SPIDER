import os
from datetime import datetime, timezone, timedelta
import json
import logging
from collections import defaultdict


def setupLogger(log_dir: str | None = None, level=logging.INFO):
    logger = logging.getLogger("SPIDER.omni2")
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logfile = f"omni2_{timestamp}.log"
        fh = logging.FileHandler(os.path.join(log_dir, logfile))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger
logger = logging.getLogger("SPIDER.omni2")


# OMNI2 uses placeholder values to indicate missing or invalid data.
# These must be converted to NaN for numerical analysis later
OMNI_FILL = {99.99, 999.9, 999.99, 9999.0}

# Mapping of OMNI2 physical parameters to column indices
# Note: OMNI2 docs use 1-based index, Python is 0-based.
FIELDS = {    
    "bz_gsm": {
        "index": 16,
        "unit": "nT", # nano-Tesla
        "description": "IMF Bz (downward) component in GSM coordinates"
        },       # Word 17 - is IMF oriented to open magnetosphere or keep it closed?
    "b_mag": {
        "index": 8,
        "unit": "nT", # nano-Tesla
        "description": "IMF Magnitude |B|"
        },      # Word 9 - how strong is the magnetic field carried by solar wind?
    "v_sw": {
        "index": 24,
        "unit": "km/s", # Kilometres per second
        "description": "Solar wind (flow) speed"
        },      # Word 25 - how fast is the solar wind hitting Earth?
    "np": {
        "index": 23,
        "unit": "cm^-3", # Number of Protons per unit volume
        "description": "Proton number Density"
        },       # Word 24 - how many particles are in the solar wind?
    "pdyn": {
        "index": 28,
        "unit": "nPa", # nano-Pascals
        "description": "Solar Wind Dynamic Pressure"
        },      # Word 29 - how hard is the solar wind pushing on Earth's magnetosphere?
    "ey": {
        "index": 35,
        "unit": "mV/m", # milli-Volt per metre
        "description": "Solar Wind Electric Field"
        },      # Word 36 - how strong is the solar wind driving energy into the magnetosphere?
    "beta": {
        "index": 36,
        "unit": "dimensionless",
        "description": "Plasma Beta (thermal to magnetic pressure ratio)"
        },      # Word 37 - is solar wind controlled more by particles or magnetic fields?
    "mach_alfven": {
        "index": 37,
        "unit": "dimensionless",
        "description": "Alfven Mach Number (solar wind speed / Alfven speed)"
        },      # Word 38 - how violent is the solar wind?
    "f10.7": {
        "index": 50,
        "unit": "sfu", # Solar Flux unit (1 sfu = 10^-22 W.m^-2.Hz^-1 )
        "description": "F10.7 solar radio flux (long-term)"
        }       # Word 51 - how active is the Sun's background energy output (quiet vs active Sun)
}

def toFloat(val: str):
    '''
    Convert OMNI2 string values to float.
    Replaces known OMNI fill values and invalid entries with NaN.
    '''
    try:
        v = float(val)
        return None if v in OMNI_FILL else v
    except ValueError:
        # Handles missing or malformed fields
        return None

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
        for name, meta in FIELDS.items():
            record[name] = {
                "value": toFloat(parts[meta["index"]]),
                "unit": meta["unit"]
                }
        
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

    logger.info(f"Dumped OMNI2 {year}-{month:02d} -> {out_file}")

def main():
    base = os.environ.get("SPIDER")
    if base is None:
        raise EnvironmentError("SPIDER system variable is not set")
    
    logger = setupLogger(
        log_dir=os.path.join(base, "logs"),
        level=logging.INFO
    )
    
    raw_path = os.path.join(base, "data/raw/nasa_omni/omni2/")
    out_path = os.path.join(base, "data/data_processed/omni2/test")

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