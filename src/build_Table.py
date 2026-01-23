import os
import json
import datetime
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

def extractObservedKp():
    pass

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
    years = sorted(d for d in os.listdir(observed_path) 
                   if os.path.isdir(os.path.join(observed_path, d)))
    for year in years:
        year_path = os.path.join(observed_path, year)

        for file_name in sorted(os.listdir(year_path)):
            if not file_name.endswith(".json"):
                continue
                
            filepath = os.path.join(year_path, file_name)
            observed_json = importFile(filepath)

            rows = extractObservedKp()

    print("STOP!")

if __name__ == "__main__":
    main()