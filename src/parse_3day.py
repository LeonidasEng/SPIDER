# Parse data from the 3 Day Forecast Text File
# Do it for one file
# Then for two files to see how files can combine
# Then create for entire time period

import os 
import re
from datetime import datetime
import json
from collections import defaultdict

def fileFetch(base_path:str):
    for root, dirs, files in os.walk(base_path):
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)
                with open(file_path, "r") as f:
                    text = f.readlines()
                yield file_path, text # Yield keyword to retrieve more than one file on iteration
 
def parseSections(text:list):
    '''
    Extract relevant sections from text files.
    
    :param text: Extract relevant section data
    :type text: list
    '''
    issue_ln = text[1].replace("Issued:", "").strip()
    issue = datetime.strptime(issue_ln, "%Y %b %d %H%M %Z")
    issue_dt = str(issue.date())
    del text[2:4] # delete comments
    
    kp_data, radiation_data, blackout_data = [], [], [] # Create lists for three_day_forecast text
    section = None
    for i, row in enumerate(text):
        line = row.strip() # Remove whitespace
        # Find the sections in the 3day forecast
        if "noaa geomagnetic activity" in line.lower():
            section = "kp" # Start Kp section
            continue
        elif "noaa solar radiation activity" in line.lower():
            section = "radiation" # Start Radiation section
            continue
        elif "noaa radio blackout activity" in line.lower():
            section = "blackout" # Start Radio Blackout section
            continue

        if not line.strip():
            continue

        if section == "kp":
            kp_data.append(line)
        elif section == "radiation":
            radiation_data.append(line)
        elif section == "blackout":
            blackout_data.append(line)

    return kp_data, radiation_data, blackout_data, issue_dt

def extractKpMeta(kp_data:list):
    '''
    Extract metadata from text-based Kp section
    
    :param kp_data: Kp section with text and numeric data.
    :return meta: meta data for Kp
    '''
    meta = {
        "greatest_observed_kp": None,
        "greatest_expected_kp": None,
        "greatest_expected_scale": None,
        "rationale": "" # Rationale can be multi-line
    }
    capture_rationale = False

    for line in kp_data:
        l = line.lower()
        if "greatest observed" in l:
            m = re.search(r"was\s+([\d.]+)", l) # regular expression for observed string (was) and float 
            if m:
                meta["greatest_observed_kp"] = float(m.group(1))
        elif "greatest expected" in l:
            m_val = re.search(r"is\s+([\d.]+)", l) # regular expression for predicted string (is) and float
            m_scale = re.search(r"\((g\d)\)", l) # regular expression for scale id
            if m_val:
                meta["greatest_expected_kp"] = float(m_val.group(1))
            if m_scale:
                meta["greatest_expected_scale"] = m_scale.group(1).upper()
        
        elif l.startswith("rationale:"):
            capture_rationale = True
            meta["rationale"] += line.replace("Rationale:", "").strip() + " " # Adding whitespace to separate lines

        elif capture_rationale:
            meta["rationale"] += line.strip() + " " # Ensure captured rationale has no leading/trailing whitespace
    
    meta["rationale"] = meta["rationale"].strip()

    return meta


def extractRadiationMeta(radiation_data:list):
    '''
    Extract metadata from text-based Solar Radiation section
    
    :param radiation_data: Solar Radiation section with text and numeric data
    :return meta: Metadata for Solar Radiation
    '''
    meta = {
        "radiation_observed": None,
        "rationale": "" # Rationale can be multi-line
    }

    capture_rationale = False

    for line in radiation_data:
        l = line.lower()

        if "was below s-scale" in l:
            meta["radiation_observed"] = False
        elif "was above s-scale" in l:
            meta["radiation_observed"] = True
        
        elif l.startswith("rationale:"):
            capture_rationale = True
            meta["rationale"] += line.replace("Rationale", "").strip() + " " # Add whitespace between lines

        elif capture_rationale:
            meta["rationale"] += line.strip() + " " # Ensure captured rationale has no leading/trailing whitespace
        
    meta["rationale"] = meta["rationale"].strip()

    return meta

def extractBlackoutMeta(blackout_data:list):
    '''
    Extract metadata from text-based Radio Blackout section
    
    :param blackout_data: Radio Blackout section with text and numeric data
    
    :return meta: Metadata for Radio Blackout data
    '''
    meta = {
        "blackout_observed": None,
        "max_blackout_level": None,
        "max_blackout_time": None,
        "rationale": "" # Rationale can be multi-line
    }

    capture_rationale = False

    for line in blackout_data:
        l = line.lower()

        if "radio blackouts reaching" in l:
            meta["blackout_observed"] = True

            m_level = re.search(r"(r\d)") # Extract Radio scale (if any) 
            m_time = re.search(r"at\s+(.*utc)", line, re.IGNORECASE) # Extract blackout time (if any)

            if m_level:
                meta["max_blackout_level"] = m_level.group(1).upper()
            if m_time:
                meta["max_blackout_time"] = m_time.group(1).strip()

        elif "no radio blackouts were observed" in l:
            meta["blackout_observed"] = False
            
        elif l.startswith("rationale:"):
            capture_rationale = True
            meta["rationale"] += line.replace("Rationale", "").strip() + " " # Add whitespace between lines
        
        elif capture_rationale:
            meta["rationale"] += line.strip() + " "  # Ensure captured rationale has no leading/trailing whitespace

    meta["rationale"] = meta["rationale"].strip()

    return meta

def cleanForecastData(kp_data:list, radiation_data:list, blackout_data:list):
    '''
    Extract the tabular data from the forecast sections
    
    :param kp_data: Kp table + text 
    :param radiation_data: Radiation table + text
    :param blackout_data: Blackout table + text
    
    :return kp_data_cleaned: Kp section with 3-hour granularity
    :return radiation_data_cleaned: Radiation probability table
    :return blackout_data_cleaned: Radio Blackout probability table
    '''
    def extractBlock(lines:list, start_text:str):
        '''
        Extract tabular data
        
        :param lines: Section text
        :param start_text: Define the start of the tabular data
        :return cleaned: tabular data
        '''
        cleaned = []
        capture = False
        for line in lines:
            l = line.lower().strip()

            if l.startswith(start_text):
                capture = True
            
            if l.startswith("rationale"):
                break
            
            if capture:
                cleaned.append(line)
        return cleaned
    
    kp_data_cleaned = extractBlock(kp_data, start_text="noaa kp index breakdown") # Extract Kp data
    radiation_data_cleaned = extractBlock(radiation_data, start_text="solar radiation storm forecast") # Extract Solar Radiation data
    blackout_data_cleaned = extractBlock(blackout_data, start_text="radio blackout forecast") # Extract Radio Blackout data

    return kp_data_cleaned, radiation_data_cleaned, blackout_data_cleaned

def buildIndices(kp, radiation, blackout):
    pass

def attachMeta():
    pass

def main():
    base = os.environ.get('SPIDER')
    if base is None:
        raise EnvironmentError("SPIDER system variable is not set!")
    data_rel = "data/raw/3day_forecast/20230202_20230205_raw"
    data_path = os.path.join(base, data_rel)
    processed_path = os.path.join(base, "data", "data_processed", "3day_forecast")
    data_dict = defaultdict(lambda: defaultdict(dict))

    for file_path, text in fileFetch(data_path):
        parts = os.path.normpath(file_path).split(os.sep)
        year = int(parts[-3])
        month = int(parts[-2])

        kp_data, radiation_data, blackout_data, issue_dt = parseSections(text)
        kp_meta = extractKpMeta(kp_data)
        radiation_meta = extractBlackoutMeta(radiation_data)
        blackout_meta = extractRadiationMeta(blackout_data)
        kp_data_cleaned, radiation_data_cleaned, blackout_data_cleaned = cleanForecastData(
            kp_data, radiation_data, blackout_data)
        forecast = buildIndices(kp_data_cleaned, radiation_data_cleaned, blackout_data_cleaned, issue_dt)
        # forecast + meta_datas
    
    #dump to JSON

        print("BREAK")
if __name__ == "__main__":
    main()
