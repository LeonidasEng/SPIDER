# Parse data from the Geomag Forecast Text File
# Do it for one file
# Then for two files to see how files can combine
# Then create for entire time period

# Simplest one, should go first.
import os
from datetime import datetime, timedelta, timezone
import json
import logging
from collections import defaultdict

def setupLogger(log_dir: str | None = None, level=logging.INFO):
    logger = logging.getLogger("SPIDER.geomag")
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
        logfile = f"geomag_{timestamp}.log"
        fh = logging.FileHandler(os.path.join(log_dir, logfile))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger
logger = logging.getLogger("SPIDER.geomag")


def fileFetch(base_path:str):
    for root, dirs, files in os.walk(base_path):
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)
                with open(file_path, "r") as f:
                    text = f.readlines()
                yield file_path, text # Yield keyword to retrieve more than one file on iteration
               
def parseSections(text:list):
    issue_ln = text[1].replace(":Issued:", "").strip()
    issue = datetime.strptime(issue_ln, "%Y %b %d %H%M %Z")
    issue = issue.replace(tzinfo=timezone.utc)
    issue_dt = issue.date().isoformat()  # e.g. '2025-08-31' - must be datetime for timedelta
    issue_ts = issue # full datetime

    del text[2:4] # delete comments
    ap_data, geomag_data, kp_data = [], [], [] # Create blank lists for text sections
    section = None
    for i, row in enumerate(text):
        line = row.strip() # Remove leading/trailing whitespace
        
        # Find the sections in the Geomag forcast (Ap, Probs, Kp)
        if "noaa ap index forecast" in line.lower():
            section = "ap" # Set Ap as current section
            continue
        elif "noaa geomagnetic activity probabilities" in line.lower():
            section = "geomag" # Set Geomag as current section
            continue
        elif "noaa kp index forecast" in line.lower():
            section = "kp" # Set Kp as current section
            continue

        if not line.strip():
            continue # Remove empty entries

        if section == "ap":
            ap_data.append(line) # Ap raw text section 
        elif section == "geomag":
            geomag_data.append(line) # Geomag raw text section
        elif section == "kp":
            kp_data.append(line) # # Kp raw text section
    return kp_data, ap_data, geomag_data, issue_dt, issue_ts
  

def buildIndices(kp_data:list, ap_data:list, geomag_data:list, issue_dt:str, issue_ts:datetime):
    forecast_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict))) # Initialise structure
    forecast_dict[issue_dt]["issue"] = issue_ts.strftime("%Y-%m-%d %H:%M:%S %Z")

    # Kp Parsing
    try:
        base_date = issue_ts.date()

        for line in kp_data[1:]:
            parts = line.split()
            time_bin = parts[0] # Only section that has 3-hour bins
                
            for i, key in enumerate(("n+1", "n+2", "n+3"), start=1): # Pass one for day-alignment
                predicted_date = (base_date + timedelta(days=i)).isoformat() # Ensures predictions align with forecast not issue date
                full_time = f"{predicted_date} {time_bin}"
                forecast_dict[issue_dt]["kp"].setdefault(key, []) # Set default cleaner than if statement
                forecast_dict[issue_dt]["kp"][key].append((full_time, float(parts[i]))) # Kp is float

    except Exception as e:
        logger.error(f"Kp build failed due to an error: {e}")

    # Ap Parsing
    try:
        for line in ap_data:
            parts = line.split()
            if not parts:
                continue
            kind = parts[0]
            if kind == "Observed":
                forecast_dict[issue_dt]["ap"]["observed"] = int(parts[-1]) # Last element
            elif kind == "Estimated":
                forecast_dict[issue_dt]["ap"]["estimated"] = int(parts[-1]) # Last element
            elif kind == "Predicted":
                values = parts[-1].split("-") # Split last element into 3
                forecast_dict[issue_dt]["ap"]["n+1"] = int(values[0])
                forecast_dict[issue_dt]["ap"]["n+2"] = int(values[1])
                forecast_dict[issue_dt]["ap"]["n+3"] = int(values[2])
    except Exception as e:
        logger.error(f"Ap build failed due to an error: {e}")

    # Geomag Parsing
    for line in geomag_data:
        parts = line.split()
        if not parts:
            continue
        storm_type = " ".join(parts[:-1]) # Everything but last element
        probs = parts[-1]
        prob_values = [int(p) for p in probs.split("/") if p.isdigit()] # Split probabilities per day
        forecast_dict[issue_dt]["geomag"]["n+1"][storm_type] = prob_values[0]
        forecast_dict[issue_dt]["geomag"]["n+2"][storm_type] = prob_values[1]
        forecast_dict[issue_dt]["geomag"]["n+3"][storm_type] = prob_values[2]
        forecast_dict[issue_dt]["geomag"]["unit"] = "%"
    return forecast_dict

def dumpJob(year:int, month:int, month_data:dict, proc_output:str):
    out_dir = os.path.join(proc_output, str(year))
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, f"geomag_{year}_{month:02d}.json")

    # Sorted by Issue Date
    ordered_month_data = dict(sorted(month_data.items(), key=lambda x: x[0]))

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ordered_month_data, f, indent=4)
    logger.info(f"Dumped data for {year}-{month:02d} -> {out_file}")

def main():
    base = os.environ.get('SPIDER')
    if base is None:
        raise EnvironmentError("SPIDER system variable is not set!")
    
    logger = setupLogger(
        log_dir=os.path.join(base, "logs"),
        level=logging.INFO
    )

    data_rel = "data/raw/forecasts/geomag/"
    data_path = os.path.join(base, data_rel)
    processed_path = os.path.join(base, "data", "data_processed", "geomag_forecast") 
    data_dict = defaultdict(lambda: defaultdict(dict))

    for file_path, text in fileFetch(data_path): # generate through files with yield 
        parts = os.path.normpath(file_path).split(os.sep) # cross-platform
        year = int(parts[-3])
        month = int(parts[-2])

        kp_data, ap_data, geomag_data, issue_dt, issue_ts = parseSections(text)
        forecast = buildIndices(kp_data, ap_data, geomag_data, issue_dt, issue_ts)
        
        data_dict[year][month].update(forecast) # Extend each month file don't override
    
    # Dump every month processed as a JSON file
    for year in sorted(data_dict):
        for month in sorted(data_dict[year]):
            dumpJob(year, month, data_dict[year][month], processed_path)


if __name__ == "__main__":
    main()
