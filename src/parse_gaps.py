import os
import json
from datetime import datetime, timezone, timedelta
import logging
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

def normaliseRecord(data: dict) -> dict:
    # Normalise each record using the known SPIDER schema
    issue = datetime.fromisoformat(data["cycle"].replace("Z", "+00:00"))
    valid = datetime.fromisoformat(data["valid"].replace("Z", "+00:00"))

    return {
        "issue_time_utc": issue,
        "valid_start_utc": valid,
        "lead_time_hrs": data["tau"],
        "forecast_kp": data["forecastKp"],
        "observed_kp": data["observedKp"]
    }

def splitJSON(threeday_json:list):
    
    # Keep as lists to allow sorting later
    threeday_0030 = []
    threeday_1230 = []

    for data in threeday_json:
        record = normaliseRecord(data)

        if record["issue_time_utc"].hour == 0:
            threeday_0030.append(record)
        elif record["issue_time_utc"].hour == 12:
            threeday_1230.append(record)
        else:
            # There should be only two issues
            raise ValueError("Unexpected issue time detected.")

    return threeday_0030, threeday_1230

def sortJSON(threeday_0030, threeday_1230):
    def sorter(record):
        return (record["issue_time_utc"], record["valid_start_utc"])
    
    sorted_0030 = sorted(threeday_0030, key=sorter) # Each record passed uses helper tuple
    sorted_1230 = sorted(threeday_1230, key=sorter)

    return sorted_0030, sorted_1230

def main():
    base = os.environ.get("SPIDER")
    if base is None:
        raise EnvironmentError("SPIDER system variable is not set!")
    
    raw_path = os.path.join(base, "data", "raw", "forecasts", "3day", "time_gaps")
    processed_path = os.path.join(base, "data", "data_processed", "3day_forecast")

    for gap, file_name in GAPS.items():
        threeday_json = importFile(os.path.join(raw_path, file_name))
        threeday_0030, threeday_1230 = splitJSON(threeday_json)
        sorted_0030, sorted_1230 = sortJSON(threeday_0030, threeday_1230)
    



if __name__ == "__main__":
    main()