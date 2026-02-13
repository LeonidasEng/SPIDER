import os 
import re
from datetime import datetime, timezone, timedelta
import json
import logging
from collections import defaultdict

def setupLogger(log_dir: str | None = None, level=logging.INFO):
    logger = logging.getLogger("SPIDER.3day")
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logfile = f"3day_{timestamp}.log"
        fh = logging.FileHandler(os.path.join(log_dir, logfile))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger
logger = logging.getLogger("SPIDER.3day")


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
    issue_ln = text[1].replace("Issued", "").replace(":", "").strip()
    issue = datetime.strptime(issue_ln, "%Y %b %d %H%M %Z")
    issue_dt = str(issue.date())
    issue_ts = issue # full datetime
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

    return kp_data, radiation_data, blackout_data, issue_dt, issue_ts

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
    
    # Join lines to avoid missing values over new lines
    joined = " ".join(line.strip() for line in kp_data)
    joined = re.sub(r"\s+", " ", joined)

    m_obs = re.search(r"greatest observed.*?was\s+([\d.]+)", joined, re.IGNORECASE) # Observed Kp
    # Capture any text in between (non-greedy), keyword "was", at least one space and capture any number with decimal
    if m_obs:
        meta["greatest_observed_kp"] = float(m_obs.group(1))

    m_exp = re.search(r"greatest expected.*?is\s+([\d.]+)", joined, re.IGNORECASE) # Expected Kp value
    # Capture any text in between (non-greedy), keyword "is", at least one space and capture any number with decimal
    if m_exp:
        meta["greatest_expected_kp"] = float(m_exp.group(1))

    m_scale = re.search(r"noaa scale\s*(g\d)", joined, re.IGNORECASE) # NOAA scale
    # Capturea any scale with at least one space, the letter "g" followed by a number
    if m_scale:
        meta["greatest_expected_scale"] = m_scale.group(1).upper()

    # Rationale capture
    rationale_lines = []
    capture = False
    for line in kp_data:
        if line.lower().startswith("rationale:"):
            capture = True
            rationale_lines.append(line.split(":", 1)[1].strip()) # Capture info not header
        elif capture:
            rationale_lines.append(line.strip())

    meta["rationale"] = " ".join(rationale_lines).strip()

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

    # Join lines to avoid missing values over new lines
    joined = " ".join(line.strip() for line in radiation_data)
    joined = re.sub(r"\s+", " ", joined)
    l_joined = joined.lower()

    if "was below s-scale" in l_joined:
        meta["radiation_observed"] = False
    elif "was above s-scale" in l_joined:
        meta["radiation_observed"] = True

    capture_rationale = False
    rationale_lines = []

    for line in radiation_data:
        l = line.lower().strip()

        if l.startswith("rationale:"):
            capture_rationale = True
            rationale_lines.append(line.split(":", 1)[1].strip()) # Capture info not header
        elif capture_rationale:
            rationale_lines.append(line.strip())

    meta["rationale"] = " ".join(rationale_lines).strip()

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
        "rationale": ""
    }

    # Join lines to avoid missing values over new lines
    joined = " ".join(line.strip() for line in blackout_data)
    joined = re.sub(r"\s+", " ", joined)

    if "radio blackouts reaching" in joined.lower():
        meta["blackout_observed"] = True

        # Extract max blackout level (e.g. R1, R2, R3)
        m_level = re.search(r"\b(r\d)\b", joined, re.IGNORECASE) 
        # Find rX (where X is int) on it's own "\b" -> word boundary (start-end)
        if m_level:
            meta["max_blackout_level"] = m_level.group(1).upper()

        m_time = re.search(r"largest was at\s+(.+?\s+utc)", joined, re.IGNORECASE)
        # Find 'largest was at' then capture everything (.) up to first (+?) 'UTC' 
        if m_time:
            meta["max_blackout_time"] = m_time.group(1).strip()

    elif "no radio blackouts were observed" in joined.lower():
        meta["blackout_observed"] = False

    capture_rationale = False
    rationale_lines = []

    for line in blackout_data:
        l = line.lower().strip()

        if l.startswith("rationale:"):
            capture_rationale = True
            rationale_lines.append(line.split(":", 1)[1].strip()) # Capture info not header
        elif capture_rationale:
            rationale_lines.append(line.strip())

    meta["rationale"] = " ".join(rationale_lines).strip()
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
        Extract tabular data helper function
        
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

def buildIndices(kp_data:list, radiation_data:list, blackout_data:list, 
                 kp_meta:list, radiation_meta:list, blackout_meta:list, 
                 issue_dt, issue_ts:datetime):
    forecast_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # Kp parsing
    try:
        base_date = issue_ts.date()

        for line in kp_data[2:]:
            raw_parts = line.split()
            parts = []
            i = 0
            while i < len(raw_parts):
                if raw_parts[i].startswith("(") and parts:
                    parts[-1] = parts[-1] + " " + raw_parts[i] # group Kp and scaled (Gx) terms
                else:
                    parts.append(raw_parts[i]) # otherwise append as normal
                i += 1
            time_bin = raw_parts[0]

            issue_ts = issue_ts.replace(tzinfo=timezone.utc) # force UTC
            forecast_dict[issue_dt]["issue"] = issue_ts.strftime("%Y-%m-%d %H:%M:%S %Z")

            # Manually set lists for Kp
            for key in ("n", "n+1", "n+2"):
                if key not in forecast_dict[issue_dt]["kp"]:
                    forecast_dict[issue_dt]["kp"][key] = []
            
            def splitKp(cell):
                if "(" in cell and ")" in cell:
                    value, scale = cell.split("(")
                    return value.strip(), scale.strip(")")
                else:
                    return cell, None
            
            kp_n_val, kp_n_scale = splitKp(parts[1])
            kp_n1_val, kp_n1_scale = splitKp(parts[2])
            kp_n2_val, kp_n2_scale = splitKp(parts[3])

            # Streamline forecast dates for n, n+1, n+2
            for day, key, val, scale in (
                (0, "n", kp_n_val, kp_n_scale),
                (1, "n+1", kp_n1_val, kp_n1_scale),
                (2, "n+2", kp_n2_val, kp_n2_scale)):
                predicted_date = (base_date + timedelta(days=day)).isoformat() # Align prediction with forecast not issue
                full_time = f"{predicted_date} {time_bin}"
                forecast_dict[issue_dt]["kp"][key].append((full_time, float(val), scale))

        forecast_dict[issue_dt]["kp"]["meta"] = kp_meta

    except Exception as e:
        logger.error(f"Kp build failed due to an error: {e}")
    
    # Radiation parsing
    try:
        line = radiation_data[2]
        label = re.split(r"\d+%", line, maxsplit=1)[0].strip() # extract label before percentage values
        values = re.findall(r"\d+%", line) # find all percentages

        forecast_dict[issue_dt]["solar_radiation"][label] = {
            "n": int(values[0][:-1]),
            "n+1": int(values[1][:-1]),
            "n+2": int(values[2][:-1]),
            "unit": "%"
        }
    
        forecast_dict[issue_dt]["solar_radiation"]["meta"] = radiation_meta

    except Exception as e:
        logger.error(f"Radiation build failed due to an error: {e}")
    
    # Radio Blackout parsing
    try:
        for line in blackout_data[2:]:
            label = re.split(r"\d+%", line, maxsplit=1)[0].strip() # extract label before percentage values
            values = re.findall(r"\d+%", line) # find all percentages

            forecast_dict[issue_dt]["radio_blackout"][label] = {
                "n": int(values[0][:-1]),
                "n+1": int(values[1][:-1]),
                "n+2": int(values[2][:-1]),
                "unit": "%"
            }
        
        forecast_dict[issue_dt]["radio_blackout"]["meta"] = blackout_meta
    
    except Exception as e:
        logger.error(f"Radio blackout build failed due to an error: {e}")
    
    return forecast_dict

def dumpJob(year:int, month:int, month_data:dict, proc_output:str):
    out_dir = os.path.join(proc_output, str(year))
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, f"3day_{year}_{month:02d}.json")

    # Sort by Issue Date
    ordered_month_data = dict(sorted(month_data.items(), key=lambda x: x[0]))

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ordered_month_data, f, indent=4)

    logger.info(f"Dumped data for {year}-{month:02d} -> {out_file}")

def classifyTimePeriods(issue_ts: datetime) -> str:
    minutes = issue_ts.hour * 60 + issue_ts.minute

    if minutes < (12 * 60 + 30):
        return "0030"
    else:
        return "1230"

def main():
    base = os.environ.get('SPIDER')
    if base is None:
        raise EnvironmentError("SPIDER system variable is not set!")
    
    logger = setupLogger(
        log_dir=os.path.join(base, "logs"),
        level=logging.INFO
    )

    data_rel = "data/raw/forecasts/3day/"
    data_path = os.path.join(base, data_rel)
    processed_path = os.path.join(base, "data", "data_processed", "3day_forecast")
    data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict))) # data_dict[slot][year][month]
    latest_issue_ts = {} # {"YYYY-MM-DD": datetime }

    for file_path, text in fileFetch(data_path):
        if not text:
            logger.error(f"EMPTY FILE: {file_path}")
            continue
        logger.debug(f"Parsing file: {file_path} | Lines: {len(text)}")

        parts = os.path.normpath(file_path).split(os.sep)
        year = int(parts[-3])
        month = int(parts[-2])

        kp_data, radiation_data, blackout_data, issue_dt, issue_ts = parseSections(text)

        issue_slot = classifyTimePeriods(issue_ts) # 0030 or 1230
        
        # Make both 0030 and 1230 available (let data drive omission decision)
        slot_key = (issue_dt, issue_slot)
        
        prev_ts = latest_issue_ts.get(slot_key)
        if prev_ts is not None and issue_ts <= prev_ts:
            logger.info(f"Skipping older {issue_slot} forecast "
                        f"{issue_ts.strftime('%H:%M')} for {issue_dt}")
            continue

        # Update latest timestamp
        latest_issue_ts[issue_dt] = issue_ts
        kp_meta = extractKpMeta(kp_data)
        radiation_meta = extractRadiationMeta(radiation_data)
        blackout_meta = extractBlackoutMeta(blackout_data)
        kp_data, radiation_data, blackout_data = cleanForecastData(kp_data, radiation_data, blackout_data)
        forecast = buildIndices(kp_data, radiation_data, blackout_data, 
                                kp_meta, radiation_meta, blackout_meta,
                                issue_dt, issue_ts)
        data_dict[issue_slot][year][month].update(forecast) # Extend each month file don't override
        
    # Dump every month processed as a JSON file
    for slot in sorted(data_dict):
        slot_path = os.path.join(processed_path, f"3day_{slot}")
    
        for year in sorted(data_dict[slot]):
            for month in sorted(data_dict[slot][year]):
                dumpJob(year, month, data_dict[slot][year][month], slot_path)

if __name__ == "__main__":
    main()
