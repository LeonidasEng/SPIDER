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

            valid_start = datetime.strptime(f"{date_part} {start_hour}", "%Y-%m-%d %H").replace(tzinfo=timezone.utc)

            rows.append({
                "valid_start_utc": valid_start,
                "kp_obs": kp_obs
            })

    return rows
            
def extractGeomagForecastKp(geomag_json):
    rows = []

    for day, forecast in geomag_json.items():
        for key, data in forecast.items():
            # Multiple keys at this level
            if key == "issue":
                # Track issue date to compute lead day and hour
                issue_date = data
                date_part, time_part, tz_part = issue_date.split() # Date Time Timezone
                start_time = time_part.rsplit(":", 1)[0] # Extract Hours and Minutes - some issues aren't on the hour
                issue_start = datetime.strptime(f"{date_part} {start_time}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                continue
            elif key != "kp":
                # Ignore for now - Reintroduce other data later in project
                continue 

            forecast_bins = ["n+1", "n+2", "n+3"] # Iterate through forecast bins
            for bin in forecast_bins:
                n_bins = data.get(bin, [])
                for bin_entry in n_bins:
                    time_str, geomag_kp = bin_entry
                    date_part, hour_part = time_str.split() # Date Time
                    start_hour = hour_part.split("-")[0] # 00-03UT get first part

                    valid_start = datetime.strptime(f"{date_part} {start_hour}", "%Y-%m-%d %H").replace(tzinfo=timezone.utc)

                    lead = valid_start - issue_start
                    lead_hours = (lead.total_seconds() / 3600) # Total seconds / Seconds per hour

                    rows.append({
                        "issue_time_utc": issue_start,
                        "valid_start_utc": valid_start,
                        "kp_geomag": geomag_kp,         # Geomag Prediction
                        "lead_day": lead.days,          # Days since issue, expect 1,2,3
                        "lead_time": lead_hours         # Hours since issue
                    })

    return rows

def extract3dayForecastKp(three_day_json:str):
    rows = []

    for day, forecast in three_day_json.items():
        for key, data in forecast.items():
            if key == "issue":
                # Track issue date to comput lead day and hour
                issue_date = data
                date_part, time_part, tz_part = issue_date.split() # Date Time Timezone
                start_time = time_part.rsplit(":", 1)[0] # Extract Hourse and Minutes - some issues aren't on the hour
                issue_start = datetime.strptime(f"{date_part} {start_time}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                continue
            elif key != "kp":
                # Ignore for now - Reintroduce other data later in project
                # Metadata could be useful later
                continue

            forecast_bins = ["n", "n+1", "n+2"] # Iterate through forecast bins
            for bin in forecast_bins:
                n_bins = data.get(bin, [])
                for bin_entry in n_bins:
                    time_str, threeday_kp, scale_kp = bin_entry
                    date_part, hour_part = time_str.split() # Date Time
                    start_hour = hour_part.split("-")[0] # 00-03UT get first part

                    valid_start = datetime.strptime(f"{date_part} {start_hour}", "%Y-%m-%d %H").replace(tzinfo=timezone.utc)

                    lead = valid_start - issue_start
                    lead_hours = (lead.total_seconds() / 3600) # Total seconds / Seconds per hour

                    if lead_hours < 0:
                        continue # Omit readings before issue date 00-12UT

                    rows.append({
                        "issue_time_utc": issue_start,
                        "valid_start_utc": valid_start,
                        "kp_threeday": threeday_kp,     # Three day Prediction
                        "lead_day": lead.days,          # Days since issue, expect 0,1,2
                        "lead_time": lead_hours         # Hours since issue
                    })

    return rows


def extractOmni2(omni2_json:dict):
    rows = []

    FIELDS = [
        "bz_gsm",
        "b_mag",
        "v_sw",
        "np",
        "pdyn",
        "ey",
        "beta",
        "mach_alfven",
        "f10.7"    
    ] # Iterate through FIELDS safer

    for day, daily_data in omni2_json.items():
        for hour, hourly_data in daily_data.items():

            valid_start = datetime.strptime(f"{day} {hour}", "%Y-%m-%d %H").replace(tzinfo=timezone.utc)
            
            row = {"valid_start_utc": valid_start,
                   **{field: hourly_data.get(field, {}).get("value") for field in FIELDS}} # Get "value"s value with (**)

            rows.append(row)
    
    return rows

def build3DayForecast(three_day_forecast_path:str):
    threeday_data = []

    years = sorted(d for d in os.listdir(three_day_forecast_path)
                   if os.path.isdir(os.path.join(three_day_forecast_path, d)))
    
    # Year sub-folders inside processed 3day_forecast
    for year in years:
        year_path = os.path.join(three_day_forecast_path, year)

        # Years are broken down into monthly files
        for file_name in sorted(os.listdir(year_path)):
            if not file_name.endswith(".json"):
                continue
            
            file_path = os.path.join(year_path, file_name)
            three_day_json = importFile(file_path)

            rows = extract3dayForecastKp(three_day_json)
            threeday_data.extend(rows)
    
    print(f"Extracted {len(threeday_data)} 3day bins")

    threeday_data.sort(key=lambda x: x["issue_time_utc"]) # sort by issue_time_utc or valid_start_utc
    df_threeday = pd.DataFrame(threeday_data) # Create DataFrame for 3day forecat
    df_threeday = df_threeday.sort_values(["issue_time_utc", "valid_start_utc"]).reset_index(drop=True) # Enforce sort and remove index to new one

    return df_threeday

def buildGeomagForecast(geomag_forecast_path:str):
    geomag_data = []

    years = sorted(d for d in os.listdir(geomag_forecast_path)
                   if os.path.isdir(os.path.join(geomag_forecast_path, d)))
    
    # Year sub-folders inside processed geomag_forecast
    for year in years:
        year_path = os.path.join(geomag_forecast_path, year)

        # Years are broken down into monthly files
        for file_name in sorted(os.listdir(year_path)):
            if not file_name.endswith(".json"):
                continue
        
            file_path = os.path.join(year_path, file_name)
            geomag_json = importFile(file_path)

            rows = extractGeomagForecastKp(geomag_json)
            geomag_data.extend(rows) # Extend data with all available forecast data
    
    print(f"Extracted {len(geomag_data)} geomag bins")

    geomag_data.sort(key=lambda x: x["issue_time_utc"])
    df_geomag = pd.DataFrame(geomag_data) # Create DataFrame for geomag forecast
    df_geomag = df_geomag.sort_values(["issue_time_utc", "valid_start_utc"]).reset_index(drop=True) # Enforce sort and remove index to set new one
    
    return df_geomag

def buildObserved(observed_path:str):
    observed_data = []
    
    # Get sorted list of year sub-directories inside observed path
    years = sorted(d for d in os.listdir(observed_path) 
                   if os.path.isdir(os.path.join(observed_path, d)))
    
    for year in years:
        year_path = os.path.join(observed_path, year)
        
        # Sort and extract files in a single year directory
        for file_name in sorted(os.listdir(year_path)):
            if not file_name.endswith(".json"):
                continue # If not JSON
            
            file_path = os.path.join(year_path, file_name)
            observed_json = importFile(file_path)

            rows = extractObservedKp(observed_json)
            observed_data.extend(rows)
    
    print(f"Extracted {len(observed_data)} observed bins")
    
    observed_data.sort(key=lambda x: x["valid_start_utc"]) # Sort by valid start
    df_obs = pd.DataFrame(observed_data) # Create DataFrame for observed Kp
    df_obs = df_obs.sort_values("valid_start_utc").reset_index(drop=True) # Enforce sort and remove index set new one

    return df_obs

def buildOMNI(omni_path:str):
    omni2_data = []

    # Get sorted list of year sub-directories inside observed
    years = sorted(d for d in os.listdir(omni_path)
                   if os.path.isdir(os.path.join(omni_path, d)))
    
    for year in years:
        year_path = os.path.join(omni_path, year)

        # Sort and extract files in a single year directory
        for file_name in sorted(os.listdir(year_path)):
            if not file_name.endswith(".json"):
                continue

            file_path = os.path.join(year_path, file_name)
            omni2_json = importFile(file_path)

            rows = extractOmni2(omni2_json)
            omni2_data.extend(rows)
    
    print(f"Extracted {len(omni2_data)} OMNI2 bins")

    omni2_data.sort(key=lambda x: x["valid_start_utc"])
    df_omni = pd.DataFrame(omni2_data)
    df_omni = df_omni.sort_values("valid_start_utc").reset_index(drop=True)

    return df_omni


def buildTable(df_obs:pd.DataFrame, df_geomag:pd.DataFrame, df_3day:pd.DataFrame, df_omni:pd.DataFrame) -> pd.DataFrame:

    df = df_obs.copy()
    # 3DAY
    df = df.merge(df_3day, how="left", on="valid_start_utc", suffixes=("", "_3Day"))
    df = df.sort_values(["issue_time_utc", "valid_start_utc"]).reset_index(drop=True)
    
    missing_cols = [
        "issue_time_utc",
        "kp_threeday",
        "lead_day",
        "lead_time"
    ]

    # Both observed and predicted need to be available
    df = df.dropna(subset=missing_cols, how="all").reset_index(drop=True)

    # GEOMAG (valid_start_utc can be forecast by different issue days, standard merge will not work)
    # Lead time depend on issue_time_utc
    df = pd.merge_asof(
        df.sort_values("valid_start_utc"),
        df_geomag.sort_values("issue_time_utc"),
        left_on="valid_start_utc",
        right_on="issue_time_utc",
        direction="backward",
        suffixes=("", "_Geomag"))
    
    print("STAHP!")
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
    
    observed_path = getProcDatapath(base, "observed_kp")
    geomag_forecast_path = getProcDatapath(base, "geomag_forecast")
    three_forecast_path = getProcDatapath(base, "3day_forecast")
    omni_path = getProcDatapath(base, "omni2")

    df_obs = buildObserved(observed_path)
    df_geomag = buildGeomagForecast(geomag_forecast_path)
    df_3day = build3DayForecast(three_forecast_path)
    df_omni = buildOMNI(omni_path)

    df_all = buildTable(df_obs, df_geomag, df_3day, df_omni)
    
    '''
        print(df_obs.head())
        print(df_obs.tail())
        print(df_obs.info())
        print(df_geomag.head())
        print(df_geomag.tail())
        print(df_geomag.info())
        print(df_3day.head())
        print(df_3day.tail())
        print(df_3day.info())
        print(df_omni.head)
        print(df_omni.tail)
        print(df_omni.info)
    '''
    
    print("FIN.")

if __name__ == "__main__":
    main()



