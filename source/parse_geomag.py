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

def folderQuery(data_str:str, data_path:str):
    fnames = os.listdir(data_path)
    for file in fnames:
        file_path = os.path.join(data_path, file)
        with open(file_path, "r"):
            rows = f.readlines()
        return rows

def parseSections(rows:list):
    product = rows[0].replace(":Product:", "").strip()
    issue_ln = rows[1].replace(":Issued:", "").strip()
    issue = datetime.strptime(issue_ln, "%Y %b %d %H%M %Z")
    issue_dt = str(issue.date())  # e.g. '2025-08-31'
    del rows[2:4] # delete comments
    ap_data, geomag_data, kp_data = [], [], [] # Create blank lists for text sections
    section = None
    for i, row in enumerate(rows):
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
  

def buildIndices(kp_data, ap_data, geomag_data, issue_dt):
    forecast_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for line in kp_data[1:]:
        parts = line.split()
        time_bin = parts[0]
        full_time = f"{issue_dt} {time_bin}"
        forecast_dict[issue_dt]["kp"]["n+1"].append((full_time, parts[1]))
        forecast_dict[issue_dt]["kp"]["n+2"].append((full_time, parts[2]))
        forecast_dict[issue_dt]["kp"]["n+3"].append((full_time, parts[3]))
        

    
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
    
    data_str = "data/raw/geomag_forecast/20250831_20250901_raw/2025/08"
    data_path = os.path.join(base, data_str)

    fnames = os.listdir(data_path)
    for file in fnames:
        file_path = os.path.join(data_path, file)
        with open(file_path, "r") as f:
            rows = f.readlines()
            product = rows[0].replace(":Product:", "").strip() # Strip won't work on it's own need to replace to remove all chars
            issue_ln = rows[1].replace(":Issued:", "").strip()
            issue = datetime.strptime(issue_ln, "%Y %b %d %H%M %Z")
            del rows[2:4] # delete comments
            ap_data, geomag_data, kp_data = [], [], [] # Create blank lists for text sections
            section = None
            for i, row in enumerate(rows):
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
            
            issue_date = str(issue.date())  # e.g. '2025-08-31'
            kp_rows = []
            for line in kp_data[1:]:  # Ignore the header line
                parts = line.split()
                if len(parts) == 4:
                    time_bin = parts[0]
                    full_time = f"{issue_date} {time_bin}"  # e.g. "2025-08-31 00-03"
                    kp_rows.append([full_time, parts[1], parts[2], parts[3]])

            kp_df = pd.DataFrame(kp_rows, columns=["Time Bins", "Kp n+1", "Kp n+2", "Kp n+3"]) # df = pandas.DF(rows, columns)
            
            # Define Ap dataframe
            ap_rows = []
            yr = issue.year
            for line in ap_data:
                parts = line.split()
                kind = parts[0]
                if kind in ("Observed", "Estimated"):
                    d = datetime.strptime(f"{parts[2]} {parts[3]} {yr}", "%d %b %Y")
                    ap_rows.append([kind, d, parts[-1]]) # type, date and value                     
                elif kind == "Predicted":
                    days = list(range(int(parts[2]), int(parts[3].split('-')[1]) + 1)) # I want between (n+1) and (n+4) so I get Day 1,2,3
                    vals = parts[-1].split('-')
                    month = parts[3][:3]
                    for day, val in zip(days, vals):
                        d = datetime.strptime(f"{day} {month} {yr}", "%d %b %Y")
                        ap_rows.append([kind, d, val])
            ap_df = pd.DataFrame(ap_rows, columns=["Type", "Date", "Ap Value"])
            
            labels = []
            for t in ap_df["Type"]:
                if t == "Observed":
                    labels.append("Ap n-1")
                elif t == "Estimated":
                    labels.append("Ap n")
                elif t == "Predicted":
                    # assign incrementally for each predicted row
                    labels.append(f"Ap n+{len([x for x in labels if x.startswith('Ap n+')]) + 1}")
            ap_df["Day Label"] = labels
            ap_day_vals = list(ap_df["Ap Value"])
            ap_day_rows = []
            for i, _ in enumerate(kp_rows):
                ap_day_rows.append(ap_day_vals) # Index is only needed to repeat for multiple days
            ap_day_df = pd.DataFrame(ap_day_rows, columns=labels)
            
            # Define Geomag dataframe
            geomag_rows = []
            for line in geomag_data:
                parts = line.split()
                
                if not parts:
                    continue
                if len(parts) == 2:
                    storm_type = parts[0]
                    probs = parts[1]
                elif len(parts) == 3:
                    storm_type = parts[0]
                    probs = parts[2]
                else:
                    continue

                # Split "10/15/40" -> [10, 15, 40]
                prob_values = [f"{int(p):02d}" for p in probs.split('/')]
                geomag_rows.append([storm_type] + prob_values)
            geomag_df = pd.DataFrame(geomag_rows, columns=["Type", "P(n+1)", "P(n+2)", "P(n+3)"])

            geomag_wide_parts = []

            for col in ["P(n+1)", "P(n+2)", "P(n+3)"]:
                subset = geomag_df[["Type", col]].set_index("Type").T
                subset.columns = [f"{t} {col}" for t in subset.columns]
                geomag_wide_parts.append(subset.reset_index(drop=True))

            geomag_wide = pd.concat(geomag_wide_parts, axis=1)
            geomag_full = pd.concat([geomag_wide] * len(ap_day_df), ignore_index=True)
            geomag_full = geomag_full.astype(int)

            
            kp_columns = [f"Time Bins", "Kp n+1"]
            ap_day_columns = ["Ap n+1"]
            geomag_n1_columns = [c for c in geomag_full.columns if "P(n+1)" in c]

            n1_df = pd.concat([
                kp_df[kp_columns].reset_index(drop=True),
                ap_day_df[ap_day_columns].reset_index(drop=True),
                geomag_full[geomag_n1_columns].reset_index(drop=True)
            ], axis=1)
            print("Geomagnetic Prediction for Issue Date + 1 day")
            print(n1_df)
            

            
            