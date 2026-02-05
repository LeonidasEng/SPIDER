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
        # Helper function for reading in JSON file
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
            
            # Datetime this measurement started from (start of 3-hour bin)
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

                    # Extract valid start for Geomag Forecast (start of 3-hour bin)
                    valid_start = datetime.strptime(f"{date_part} {start_hour}", "%Y-%m-%d %H").replace(tzinfo=timezone.utc)

                    lead = valid_start - issue_start           # Calculate lead time
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
                # Track issue date to compute lead day and hour
                issue_date = data
                date_part, time_part, tz_part = issue_date.split() # Date Time Timezone
                start_time = time_part.rsplit(":", 1)[0] # Extract Hours and Minutes - some issues aren't on the hour
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

                    lead = valid_start - issue_start # Starting at Day n @ 12:30 for n-n+1-n+2 there will be 8, 8, 3 bins
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
                   **{field: hourly_data.get(field, {}).get("value") for field in FIELDS}} # Unpack value's value with (**)

            rows.append(row)
    
    return rows

def build3DayForecast(three_day_forecast_path:str):
    threeday_data = []

    # Sort if sub-folders in a directory at path
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
            
            # Extract records from JSON and store in rows
            rows = extract3dayForecastKp(three_day_json)
            threeday_data.extend(rows) # Extend data on iteration
    
    print(f"Extracted {len(threeday_data)} 3day bins")

    threeday_data.sort(key=lambda x: x["issue_time_utc"]) # sort by issue_time_utc or valid_start_utc
    df_threeday = pd.DataFrame(threeday_data) # Create DataFrame for 3day forecat
    df_threeday = df_threeday.sort_values(["issue_time_utc", "valid_start_utc"]).reset_index(drop=True) # Primary & secondary sort, reset index

    return df_threeday

def buildGeomagForecast(geomag_forecast_path:str):
    geomag_data = []

    # Sort if sub-folders in a directory at path
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
    df_geomag = df_geomag.sort_values(["issue_time_utc", "valid_start_utc"]).reset_index(drop=True) # Primary & secondary sort, reset index
    
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
    df_obs = df_obs.sort_values("valid_start_utc").reset_index(drop=True) # Primary & secondary sort, reset index

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
    
    omni2_data.sort(key=lambda x: x["valid_start_utc"]) # Sort by valid start
    df_omni = pd.DataFrame(omni2_data) # Create DataFrame for omni2
    df_omni = df_omni.sort_values("valid_start_utc").reset_index(drop=True)

    return df_omni

def buildTable3Day(df_3day:pd.DataFrame, df_obs:pd.DataFrame, df_omni:pd.DataFrame) -> pd.DataFrame:
    df = df_3day.copy() # Important: Table forecast-centric NOT observation-centric

    # 3DAY (attach valid time match)
    df = df.merge(df_obs[["valid_start_utc", "kp_obs"]], # Align with valid time, merge kp_obs
                  how="left", on="valid_start_utc")
    
    # Drop rows without valid 3-day forecast
    df = df.dropna(subset=["issue_time_utc", "kp_threeday"])

    # Merge OMNI (Upstream context (backwards), tolerance = 3 hours)
    df = pd.merge_asof(df.sort_values("valid_start_utc"),
                       df_omni.sort_values("valid_start_utc"),
                       on="valid_start_utc",
                       direction="backward",
                       tolerance=pd.Timedelta("3h"))

    df["lead_time"] = df["lead_time"].round(2) # Round to 2 decimal places 
    
    return df.sort_values(["issue_time_utc", "valid_start_utc"]).reset_index(drop=True)

def buildTableGeomag(df_geomag:pd.DataFrame, df_obs:pd.DataFrame, df_omni:pd.DataFrame) -> pd.DataFrame:
    df = df_geomag.copy() # Important: Table forecast-centric NOT observation-centric

    # Merge observed Kp (exact valid time match)
    df = df.merge(df_obs[["valid_start_utc", "kp_obs"]], how="left", on="valid_start_utc")

    # Merge OMNI (contextual)
    df = pd.merge_asof(df.sort_values("valid_start_utc"),
                       df_omni.sort_values("valid_start_utc"),
                       on="valid_start_utc",
                       direction="backward",
                       tolerance=pd.Timedelta("3h"))

    # Sanity check: no lead time leakage
    computed = ((df["valid_start_utc"] - df["issue_time_utc"]).dt.total_seconds() / 3600)
    
    assert (computed - df["lead_time"]).abs().max() < 1e-6, \
            "Lead time mismatch"
    
    df["lead_time"] = df["lead_time"].round(2) # Round to 2 decimal places 
    
    return df.sort_values(["issue_time_utc", "lead_day", "valid_start_utc"]).reset_index(drop=True)

def getProcDatapath(base:str, dataset_key: str, sub_folder: str | None = None):
    # Added subfolder param for 3day 0030 and 1230
    try:
        base_path = os.path.join(base, "data", "data_processed", DATASETS[dataset_key])
        if sub_folder:
            base_path = os.path.join(base_path, sub_folder)
        return base_path

    except:
        raise ValueError(f"Unknown dataset key: {dataset_key}")
    
def verifyDataQuality(df: pd.DataFrame, name:str="dataset") -> dict:
    # CRISP-DM Data Understanding (2.4)
    # Examine quality of data, addressing data completeness 
    # data errors, missing values

    def percent(x):
        return round(x * 100, 1)
    
    def precision(x):
        return None if pd.isna(x) else round(float(x), 2)
    
    report = {}

    report["dataset"] = name
    report["rows"] = int(len(df))
    report["columns"] = int(df.shape[1])
    
    if "valid_start_utc" in df.columns:
        df_sorted = df.sort_values("valid_start_utc")

        # Identify time range
        report["time_start"] = df_sorted["valid_start_utc"].min()
        report["time_end"] = df_sorted["valid_start_utc"].max()

        # Identify time gaps looking for differences between rows
        differences = (df_sorted["valid_start_utc"]
                       .diff()
                       .dropna()
                       .dt.total_seconds() / 3600)
        
        # Build dictionary of gaps
        report["hour_gap_distribution"] = {
            precision(k): int(v) for k, v in differences.value_counts().to_dict().items()
        }
        # Which time steps not spaced by 3 hours?
        report["time_gaps"] = int((differences != 3).sum())
        report["average_time_gaps_%"] = percent((differences !=3).mean())
    
    missing_values = df.isna().mean()
    report["missing_values_%"] = {
        col: percent(val) for col, val in missing_values.items() if val > 0
    }

    if "lead_time" in df.columns:
        report["negative_lead_times"] = int((df["lead_time"] < 0).sum())
        report["max_lead_time_hours"] = precision(df["lead_time"].max())
        report["min_lead_time_hours"] = precision(df["lead_time"].min())

    if "valid_start_utc" in df.columns:
        # Multiplicity expected
        dup = df["valid_start_utc"].duplicated().sum()
        report["duplicate_times"] = int(dup)
        report["duplicate_times_%"] = percent(dup / len(df))

    return report

def main():
    # Environment variable must be set to run this script
    base = os.environ.get("SPIDER")
    if base is None:
        raise EnvironmentError("SPIDER system variable is not set!")
    
    # Does path exist for processed data (forecasts, observed, omni)
    observed_path = getProcDatapath(base, "observed_kp")
    geomag_forecast_path = getProcDatapath(base, "geomag_forecast")
    three_forecast_morn_path = getProcDatapath(base, "3day_forecast", "3day_0030")
    three_forecast_aft_path = getProcDatapath(base, "3day_forecast", "3day_1230")
    omni_path = getProcDatapath(base, "omni2")

    # Build DataFrames for processed data
    df_obs = buildObserved(observed_path)
    df_geomag = buildGeomagForecast(geomag_forecast_path)
    df_3day_morn = build3DayForecast(three_forecast_morn_path)
    df_3day_aft = build3DayForecast(three_forecast_aft_path)
    df_omni = buildOMNI(omni_path)

    # Merge into forecast-centric DataFrames
    spider_3day_morn = buildTable3Day(df_3day_morn, df_obs, df_omni)
    spider_3day_aft = buildTable3Day(df_3day_aft, df_obs, df_omni)
    spider_geomag = buildTableGeomag(df_geomag, df_obs, df_omni)

    output_path = os.path.join(base, "data", "datasets")
    os.makedirs(output_path, exist_ok=True)

    path_3day_morn = os.path.join(output_path, "spider_features_3day_0030.parquet")
    path_3day_aft = os.path.join(output_path, "spider_features_3day_1230.parquet")
    path_geomag = os.path.join(output_path, "spider_features_geomag.parquet")
    
    with open(os.path.join(output_path, "spider_quality_report.json"), "w") as f:
        json.dump({
            "3day_0030": verifyDataQuality(spider_3day_morn, "3day_0030"),
            "3day_1230": verifyDataQuality(spider_3day_aft, "3day_1230"),
            "geomag": verifyDataQuality(spider_geomag, "geomag")
            }, f, default=str, indent=4)

    # Output merged dataframes as parquet
    spider_3day_morn.to_parquet(path_3day_morn) 
    spider_3day_aft.to_parquet(path_3day_aft)
    spider_geomag.to_parquet(path_geomag)

    

if __name__ == "__main__":
    main()



