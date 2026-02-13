import os
import json
from datetime import datetime, timedelta
from collections import defaultdict


# The kind people at SWPC provided me with this data to fill in the gaps that I found within the NCEI archive
# using an in-house tool. This data is not definitive and and may contain errors.

GAPS = {
    "Gap_2022":"20220816-20221112_KpFcst.json",
    "Gap_2023":"20230809-20240229_KpFcst.json",
    "Gap_2024":"20240901-20240930_KpFcst.json"
}

def importFile(file_path:str):
    try:
        # Helper function for reading in JSON file
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        # Should not occur as only 3 files, but good practice
        raise ValueError(f"JSON file not found at: {file_path}.")
    
def classifyStorm(kp:float):
    ''' Convert Kp to NOAA G-scale '''
    if kp >= 9: return "G5"
    if kp >= 8: return "G4"
    if kp >= 7: return "G3"
    if kp >= 6: return "G2"
    if kp >= 5: return "G1"
    return None

def normaliseRecord(data: dict) -> dict:
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
    # Keep as lists to allow sorting later
    threeday_0030 = []
    threeday_1230 = []

    for data in threeday_json:
        record = normaliseRecord(data)

        issue_date, issue_time = record["issue_time_utc"].split()
        hour = int(issue_time.split(":")[0])
        # Extract hour from time 
        if  hour < 2:
            threeday_0030.append(record)
        elif hour >= 12:
            threeday_1230.append(record)
        else:
            # There should be only two issues
            raise ValueError("Unexpected issue time detected.")

    return threeday_0030, threeday_1230

def sortJSON(threeday_0030, threeday_1230):
    # Key helper function making sorting neater
    def sorter(record):
        return (record["issue_time_utc"], record["valid_start_utc"])
    
    sorted_0030 = sorted(threeday_0030, key=sorter) # Each record passed uses helper tuple
    sorted_1230 = sorted(threeday_1230, key=sorter)

    return sorted_0030, sorted_1230

def buildForecastStruct(sorted_records:list):
    '''
    Build same structure as other parsing scripts to insert into existing
    data pipeline.
    '''
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
    # Changed to 3day_ folder to match other processed data
    out_dir = os.path.join(proc_output, "time_gaps", f"3day_{tag}")

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
    base = os.environ.get("SPIDER")
    if base is None:
        raise EnvironmentError("SPIDER system variable is not set!")
    
    raw_path = os.path.join(base, "data", "raw", "forecasts", "3day", "time_gaps")
    processed_path = os.path.join(base, "data", "data_processed", "3day_forecast")

    for _, file_name in GAPS.items():
        threeday_json = importFile(os.path.join(raw_path, file_name))
        threeday_0030, threeday_1230 = splitJSON(threeday_json)
        sorted_0030, sorted_1230 = sortJSON(threeday_0030, threeday_1230)
        # Call dump job for each issue type
        dumpJob(sorted_0030, processed_path, "0030")
        dumpJob(sorted_1230, processed_path, "1230")



if __name__ == "__main__":
    main()