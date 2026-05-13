import os
from datetime import datetime, timezone, timedelta
import json
import logging
from collections import defaultdict

def setupLogger(log_dir: str | None = None, level=logging.INFO):
    """
    Configure logger for the FTP access utility.

    The logger outputs messages to the console and, 
    if a log directory is provided stores FTP download requests 
    in a timestamped log file.

    Args:
        log_dir: Optional directory where the log file will be saved.
        level: Logging level used for console and file handler.
    
    Returns:
        logging.Logger: Configured logger for FTP access.
    """
    logger = logging.getLogger("SPIDER.dayind")
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
        logfile = f"dayind_{timestamp}.log"
        fh = logging.FileHandler(os.path.join(log_dir, logfile))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger
logger = logging.getLogger("SPIDER.dayind")


def fileFetch(base_path:str):
    """
    Recursively fetches all text files from directory and yields their file paths
    and contents.

    Args:
        base_path (str): Root directory to search for files
    
    Yields:
        tuple[str, list[str]]:
            - file_path: Path to discovered text file.
            - text: List of read lines from file.
    """
    for root, _, files, in os.walk(base_path):
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)
                with open(file_path, "r") as f:
                    text = f.readlines()
                yield file_path, text

def parseDayInd(text: list):
    """
    Parses NOAA daily geomagnetic indices (`dayind`) product text and extracts
    Kp values.

    Args:
        text (list): Raw NOAA `dayind` product text as a list of lines.
    
    Returns:
        tuple[str, list]
            - observed_dt: Observed date associated with Kp measurements.
            - kp_bins: 3-hour UTC time bins, corresponding observed Kp value.
    Notes:
        NOAA `dayind` products are issued after the observation day, therefore
        Kp measurements are associated with:
        `issue_date - 1 day`
    """
    # Extract issue timestamp from product header
    if not text:
        return None
    issue_ln = text[1].replace(":Issued:", "").strip()
    # Replace UTC or UT
    issue = datetime.strptime(issue_ln.replace(" UTC", "").replace(" UT", ""), "%Y %b %d %H%M")
    issue = issue.replace(tzinfo=timezone.utc)
    
    # NOAA dayind products are issued after the observation day
    # Kp values are associated with (issue date - 1 day)
    observed_date = (issue - timedelta(days=1)).date()
    observed_dt = str(observed_date)

    kp_bins = [] # Contains (timestamp, Kp) tuple

    is_estimated = False
    is_planetary = False # Flags to ensure Kp values are selected
    
    for line in text:
        l = line.lower().strip()

        # Ensure correct line in raw data is selected under "Estimated"
        if "estimated" in l:
            is_estimated = True
            continue
        
        if is_estimated and "planetary" in l:
            is_planetary = True
            continue
    
        if is_planetary:
            # Skip empty lines and comments
            if not l or l.startswith('#'):
                continue

            # Data line starts with A index then 8 Kp values
            parts = l.split()
            if len(parts) >= 18:
                # Extract planetary indices section
                planetary_A = parts[9] # Unused
                kp_vals = parts[10:18]

                # Fixed 3-hour UT bins used by NOAA forecasts
                bins = [
                    "00-03UT", "03-06UT", "06-09UT", "09-12UT",
                    "12-15UT", "15-18UT", "18-21UT", "21-00UT"
                ]

                # Pair each Kp value with it's corresponding 3-hour time bin
                for b, v in zip(bins, kp_vals):
                    kp_bins.append((f"{observed_dt} {b}", float(v)))
                
                break # Only one Kp row exists per file
    
    return observed_dt, kp_bins

def buildIndices(observed_dt: str, kp_bins: list):
    """
    Builds a dictionary for a single day of observed Kp data

    Args:
        observed_dt (str): Observed date associated with the Kp measurements.
        kp_bins (list): List of tuples containing 3-hour UTC time bin 
        and observed Kp value.

    Returns:
        collections.defaultdict: Dictionary containing chronologically sorted
        observed Kp values for the observation date.
    """
    out = defaultdict(dict)

    kp_bins_sorted = sorted(kp_bins, key=lambda x: x[0]) # Sort by observed date

    out[observed_dt] = {
        "estimated_planetary": kp_bins_sorted
    }
    return out

def dumpJob(year: int, month: int, month_data: dict, proc_output: str):
    """
    Writes processed monthly observed data to a JSON file.

    Args:
        year (int): Observed year used for output directory naming.
        month (int): Observed month used for each output file naming.
        month_data (dict): Dictionary containing observed data for the month.
        proc_output (str): Root directory for processed JSON files.
    
    Returns:
        None
    """
    # Create year-level output directory if it does not exist
    out_dir = os.path.join(proc_output, str(year))
    os.makedirs(out_dir, exist_ok=True)

    # Monthy output file
    out_file = os.path.join(out_dir, f"dayind_{year}_{month:02d}.json")

    # Ensure days in a month are sorted chronologically
    ordered_month_data = dict(sorted(month_data.items()))

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ordered_month_data, f, indent=4)
    
    logger.info(f"Dumped data for {year}-{month:02d} -> {out_file}")

def main():
    """
    Main entry point for processing NOAA `dayind` observed geomagnetic
    activity.

    The pipeline:
    - Locates raw `dayind` product files.
    - Extracts observed Kp measurements.
    - Builds structured daily observed dictionaries.
    - Organises observations by year and month.
    - Writes processed monthly observed data to JSON files.

    Notes:
        Raw NOAA observation files are read from:        
        `data/raw/observed_indices/observed/`
        
        Processed monthly JSON files are written to:
        `data/data_processed/dayind/`
    
    Raises:
        EnvironmentError: Raised if the `SPIDER` environment variable is not defined.

    """
    # Project path controlled by environment variable
    base = os.environ.get("SPIDER")
    if base is None:
        raise EnvironmentError("SPIDER system variable is not set!")
    
    logger = setupLogger(
        log_dir=os.path.join(base, "logs"),
        level=logging.INFO
    )
    
    # Raw data input directory - as seen from ftp_access output
    data_rel = "data/raw/observed_indices/observed/"
    data_path = os.path.join(base, data_rel)

    processed_path = os.path.join(base, "data", "data_processed", "dayind")

    # Nested structure: year -> month -> day -> Kp data
    data_dict = defaultdict(lambda: defaultdict(dict))

    for file_path, text in fileFetch(data_path):
        # Extract year and month from directory
        parts = os.path.normpath(file_path).split(os.sep)
        year = int(parts[-3])
        month = int(parts[-2])

        # Parse file and integrate into year/month
        parsed = parseDayInd(text)
        if parsed is None:
            logger.warning(f"Skipping empty or missing product {file_path}")
            continue

        issue_dt, kp_bins = parsed
        day_data = buildIndices(issue_dt, kp_bins)

        data_dict[year][month].update(day_data)
    
    # Write monthly JSON
    for year in sorted(data_dict):
        for month in sorted(data_dict[year]):
            dumpJob(year, month, data_dict[year][month], processed_path)

if __name__ == "__main__":
    main()