# Parse data from the Geomag Forecast Text File
# Do it for one file
# Then for two files to see how files can combine
# Then create for entire time period

# Simplest one, should go first.
import os
import re
from datetime import datetime
import csv
import pandas as pd
from collections import defaultdict

def fileFetch(data_path:str):
    fnames = os.listdir(data_path)
    for file_name in fnames:
        file_path = os.path.join(data_path, file_name)
        with open(file_path, "r") as f:
            text = f.readlines()
        return text

def parseSections(text:list):
    issue_ln = text[1].replace(":Issued:", "").strip()
    issue = datetime.strptime(issue_ln, "%Y %b %d %H%M %Z")
    issue_dt = str(issue.date())  # e.g. '2025-08-31'
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
    return kp_data, ap_data, geomag_data, issue_dt
  

def buildIndices(kp_data:list, ap_data:list, geomag_data:list, issue_dt:str):
    forecast_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict))) # Initialise structure
    try:
        for line in kp_data[1:]:
            parts = line.split()
            time_bin = parts[0] # Only section that has 3-hour bins
            full_time = f"{issue_dt} {time_bin}" # Main bin for future dataset
            
            # Manually set lists for Kp
            for key in ("n+1", "n+2", "n+3"):
                if key not in forecast_dict[issue_dt]["kp"]:
                    forecast_dict[issue_dt]["kp"][key] = []

            forecast_dict[issue_dt]["kp"]["n+1"].append((full_time, parts[1])) 
            forecast_dict[issue_dt]["kp"]["n+2"].append((full_time, parts[2]))
            forecast_dict[issue_dt]["kp"]["n+3"].append((full_time, parts[3]))
    except Exception as e:
        print(f"Kp build failed due to an error: {e}")
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
        print(f"Ap build failed due to an error: {e}")
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
    return forecast_dict

def dumpJob(forecast_dict):
    pass

def concatN1():
    pass

def concatN2():
    pass

def concatN3():
    pass

def main():
    pass

if __name__ == "__main__":
    base = os.environ.get('SPIDER')
    if base is None:
        raise EnvironmentError("SPIDER system variable is not set!")
    
    data_str = "data\\raw\\geomag_forecast\\20250831_20250831_raw\\2025\\08"
    data_path = os.path.join(base, data_str)

    file = fileFetch(data_path)
    kp_data, ap_data, geomag_data, issue_dt = parseSections(file)
    forecast_dict = buildIndices(kp_data, ap_data, geomag_data, issue_dt)
    print("boop!")
    
            

            
            