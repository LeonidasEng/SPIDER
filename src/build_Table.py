import os
import json
import datetime
import pandas as pd

def importFile(file_path:str):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        raise ValueError(f"JSON file not found at: {file_path}.")

def buildForecast(data:dict):


    pass

def buildObserved():
    pass

def buildOMNI():
    pass

def buildDF():
    pass

def main():
    base = os.environ.get("SPIDER")
    if base is None:
        raise EnvironmentError("SPIDER system variable is not set!")
    
    proc_path = "data/data_processed/"
    data_names = ["3day_forecast", "dayind", "geomag_forecast", "omni2"]

if __name__ == "__main__":
    main()