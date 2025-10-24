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


def extractDatefromFolder(folder:str):
    pattern = re.search(r"\d{8}_\d{8}", folder)
    if pattern:
        return datetime.strptime(pattern(1), "%Y%m%d")
    return None

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
            del rows[2:4]
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
            
            # Define Kp dataframe - 
            kp_rows = []
            for line in kp_data[1:]: # Ignore dates
                parts = line.split()
                if len(parts) == 4:
                    kp_rows.append(parts)    
            kp_df = pd.DataFrame(kp_rows, columns=["Time Bins", "Kp n+1", "Kp n+2", "Kp n+3"]) # df = pandas.DF(rows, columns)
            
            ap_rows = []
            yr = issue.year
            for line in ap_data:
                parts = line.split()
                kind = parts[0]
                if kind in ("Observed", "Estimated"):
                    d = datetime.strptime(f"{parts[2]} {parts[3]} {yr}", "%d %b %Y")
                    ap_rows.append([kind, d, parts[-1]])                     
                elif kind == "Predicted":
                    days = list(range(int(parts[2]), int(parts[3].split('-')[1]) + 1)) # I want between (n+1) and (n+4) so I get Day 1,2,3
                    vals = parts[-1].split('-')
                    month = parts[3][:3]
                    for day, val in zip(days, vals):
                        d = datetime.strptime(f"{day} {month} {yr}", "%d %b %Y")
                        ap_rows.append([kind, d, val])
            ap_df = pd.DataFrame(ap_rows, columns=["Type", "Date", "Ap Value"])
            print(kp_df)
            print("\n")
            print(ap_df)
            
            