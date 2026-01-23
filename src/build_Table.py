import os
import json
from datetime import datetime, timezone
import pandas as pd

DATASETS = {
    "observed_kp":"dayind",
    "geomag_forecast": "geomag_forecast",
    "3day_forecast": "3day_forecast",
    "omni2":"omni2"
}

def importFile(file_path:str):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        raise ValueError(f"JSON file not found at: {file_path}.")

def extractObservedKp(observed_json:dict):
    rows = []
    
    for day, data in observed_json.items():
        bins = data.get("estimated_planetary", [])

        for bin_entry in bins:
            time_str, kp_obs = bin_entry

            # 2022-01-01 00-03UT
            date_part, hour_part = time_str.split()
            start_hour = hour_part.split("-")[0]

            valid_start = datetime.strptime(f"{date_part} {start_hour}", "%Y-%M-%d %H").replace(tzinfo=timezone.utc)

            rows.append({
                "valid_start_utc": valid_start,
                "kp_obs": kp_obs
            })

    return rows
            


def extractGeomagForecast():
    pass

def extract3dayForecast():
    pass

def extractOmni2():
    pass


def buildForecast(data:dict):


    pass

def buildObserved():
    # import observed day
    
    pass

def buildOMNI():
    pass

def buildDF():
    pass


def getProcDatapath(base:str, dataset_key: str):
    try:
        return os.path.join(base, "data/data_processed", DATASETS[dataset_key])

    except:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

def main():
    base = os.environ.get("SPIDER")
    if base is None:
        raise EnvironmentError("SPIDER system variable is not set!")
    
    observed_data = {}
    observed_path = getProcDatapath(base, "observed_kp")

    all_rows = []

    years = sorted(d for d in os.listdir(observed_path) 
                   if os.path.isdir(os.path.join(observed_path, d)))
    for year in years:
        year_path = os.path.join(observed_path, year)

        for file_name in sorted(os.listdir(year_path)):
            if not file_name.endswith(".json"):
                continue
                
            filepath = os.path.join(year_path, file_name)
            observed_json = importFile(filepath)

            rows = extractObservedKp(observed_json)
            all_rows.extend(rows)
    
    print(f"Extracted {len(all_rows)} observed bins")
    all_rows.sort(key=lambda x: x["valid_start_utc"])
    df_obs = pd.DataFrame(all_rows)
    df_obs = df_obs.sort_values("valid_start_utc").reset_index(drop=True)
    print(df_obs.head())
    print(df_obs.tail())
    print(df_obs.info())

if __name__ == "__main__":
    main()



