import json
import os
from pathlib import Path
from datetime import datetime

DEFAULT_CRON_STATUS = {
    "daily_status_reset": "Scheduled",
    "sec_ticker_cik_map": "Scheduled",
    "sec_rss_filings": "Scheduled",
    "sec_xbrl_facts": "Scheduled",
    "daily_fundamentals": "Scheduled",
    "fundamental_retry": "Standby",
    "universe_regeneration": "Scheduled",
    "retention_trim": "Scheduled",
    "monthly_cold_backup": "Scheduled",
    "daily_notification": "Scheduled"
}

def update_status(updates: dict, base_path: str | Path | None = None):
    """
    Atomically updates status.json with the provided dictionary of key-value pairs.
    Performs nested dictionary merging for keys like 'cron_status' and 'u00_asset_breakdown'.
    Attaches a 'last_updated_at' timestamp.
    """
    if base_path is None:
        base_path = Path(__file__).resolve().parent.parent
    status_file = Path(base_path) / 'status.json'
    tmp_file = Path(base_path) / 'status.json.tmp'
    try:
        if status_file.exists() and status_file.stat().st_size > 0:
            try:
                with open(status_file, 'r') as f:
                    data = json.load(f)
            except Exception:
                data = {}
        else:
            data = {}
            
        for k, v in updates.items():
            if isinstance(v, dict) and isinstance(data.get(k), dict):
                data[k].update(v)
            else:
                data[k] = v
                
        data['last_updated_at'] = datetime.now().isoformat()
        with open(tmp_file, 'w') as f:
            json.dump(data, f, indent=4)
        tmp_file.replace(status_file)
    except Exception as e:
        print(f"Error updating status.json: {e}")

def update_cron_status(job_key: str, status: str = "Completed", base_path: str | Path | None = None):
    """
    Updates the execution status for a specific scheduled cron job in status.json.
    """
    update_status({"cron_status": {job_key: status}}, base_path=base_path)

def reset_daily_status(base_path: str | Path | None = None):
    """
    Executes daily at 00:00 to reset cron pipeline operational statuses and cycle EDGAR filing counters.
    - Sets yesterday's filings to today's count, and resets today's count to 0.
    - Resets fundamental_data_pulled_today to False.
    - Resets all cron operations to Scheduled/Standby.
    - Clears universes/todays_filing_symbols.json.
    """
    if base_path is None:
        base_path = Path(__file__).resolve().parent.parent
    status_file = Path(base_path) / 'status.json'
    tmp_file = Path(base_path) / 'status.json.tmp'
    try:
        if status_file.exists() and status_file.stat().st_size > 0:
            try:
                with open(status_file, 'r') as f:
                    data = json.load(f)
            except Exception:
                data = {}
        else:
            data = {}

        # Shift today's filings count to yesterday's filings count, and reset today's to 0
        current_today_filings = data.get('edgar_filings_symbols_today', 0)
        if current_today_filings == 0 and 'edgar_filings_symbols_today' not in data:
            current_today_filings = data.get('edgar_filings_symbols_yesterday', 0)

        data['edgar_filings_symbols_yesterday'] = current_today_filings
        data['edgar_filings_symbols_today'] = 0

        # Reset fundamental collection status
        data['fundamental_data_pulled_today'] = False

        # Reset all cron routine statuses back to Scheduled/Standby
        data['cron_status'] = DEFAULT_CRON_STATUS.copy()
        data['last_updated_at'] = datetime.now().isoformat()

        with open(tmp_file, 'w') as f:
            json.dump(data, f, indent=4)
        tmp_file.replace(status_file)

        # Clear todays_filing_symbols.json cache file
        todays_filings_file = Path(base_path) / 'universes' / 'todays_filing_symbols.json'
        if todays_filings_file.exists():
            with open(todays_filings_file, 'w') as f:
                json.dump([], f)
    except Exception as e:
        print(f"Error executing reset_daily_status: {e}")

def setup_dir_structure():
    """
    Sets up the directory structure for this project not found in Github (e.g. Data Folder).
    """
    base_path = Path(__file__).resolve().parent.parent
    dirs = ['data', 'logs', 'secrets', 'universes', 'tests']
    for dir_name in dirs:
        dir_path = Path(base_path) / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"    Created directory: {dir_path}")

    file_path = Path(base_path) / 'status.json'
    if not file_path.exists():
        file_path.touch()
        with open(file_path, 'w') as f:
            f.write('{}')

    print(f"    Updated file: {file_path}")