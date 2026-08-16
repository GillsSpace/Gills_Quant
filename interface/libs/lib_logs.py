import os
import re
import glob
from datetime import datetime

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')

def get_cron_pipeline_matrix():
    """
    Returns the schedule matrix of background cron jobs and raw execution log content.
    """
    temp_cron_file = os.path.join(LOGS_DIR, 'temp_cron_log.log')
    cron_log_content = ""
    if os.path.exists(temp_cron_file):
        try:
            with open(temp_cron_file, 'r') as f:
                lines = f.readlines()
                # Display up to last 40 lines to keep console view compact and readable
                cron_log_content = "".join(lines[-40:])
        except Exception as e:
            cron_log_content = f"Error reading log: {e}"

    now = datetime.now()
    cur_m = now.hour * 60 + now.minute

    # Read cron_status from status.json
    status_file = os.path.join(os.path.dirname(LOGS_DIR), 'status.json')
    cron_status = {}
    if os.path.exists(status_file):
        try:
            import json
            with open(status_file, 'r') as f:
                s_data = json.load(f)
                cron_status = s_data.get('cron_status', {})
        except Exception:
            cron_status = {}

    def get_job_status(event_key, job_time_str, fallback_time_key=None):
        if job_time_str == '5-Min Cadence':
            return 'Active'
        if '16:00' in job_time_str:
            return 'Active' if (9*60+30 <= cur_m <= 16*60) and now.weekday() < 5 else 'Standby'
        
        if event_key and event_key in cron_status:
            return cron_status[event_key]
        if fallback_time_key and fallback_time_key in cron_status:
            return cron_status[fallback_time_key]

        try:
            parts = job_time_str.split(' ')[0].split(':')
            h, m = int(parts[0]), int(parts[1])
            if 'PM' in job_time_str and h < 12:
                h += 12
            job_m = h * 60 + m
            return 'Completed' if cur_m >= job_m else 'Scheduled'
        except Exception:
            return 'Scheduled'

    pipeline_jobs = [
        {
            'time': '00:00 AM',
            'event': 'daily_status_reset',
            'name': 'Daily System & Status Reset',
            'script': 'lib_files.reset_daily_status()',
            'description': 'Resets daily cron pipeline matrix to Scheduled, cycles EDGAR filing counters, and resets daily status flags.',
            'frequency': 'Daily at 00:00 AM',
            'status': get_job_status('daily_status_reset', '00:00 AM', '00:00')
        },
        {
            'time': '03:15 AM',
            'event': 'sec_ticker_cik_map',
            'name': 'SEC Ticker → CIK Mapping Refresh',
            'script': 'lib_edgar.update_ticker_cik_map()',
            'description': 'Updates official SEC EDGAR Ticker-to-CIK JSON mapping file.',
            'frequency': 'Daily at 03:15 AM',
            'status': get_job_status('sec_ticker_cik_map', '03:15 AM', '03:15')
        },
        {
            'time': '03:20 AM',
            'event': 'sec_rss_filings',
            'name': 'SEC Daily RSS Filing Detection',
            'script': 'lib_edgar.detect_todays_filing_symbols()',
            'description': 'Scans SEC RSS submissions feed for 10-K, 10-Q, and 8-K filings from yesterday.',
            'frequency': 'Daily at 03:20 AM',
            'status': get_job_status('sec_rss_filings', '03:20 AM', '03:20')
        },
        {
            'time': '03:25 AM',
            'event': 'sec_xbrl_facts',
            'name': 'SEC XBRL Company Facts Download',
            'script': 'lib_edgar.update_current_edgar_data_file()',
            'description': 'Downloads raw XBRL company facts and builds local Parquet fundamental cache.',
            'frequency': 'Daily at 03:25 AM',
            'status': get_job_status('sec_xbrl_facts', '03:25 AM', '03:25')
        },
        {
            'time': '04:00 AM',
            'event': 'daily_fundamentals',
            'name': 'Daily Fundamental & Corporate Actions Ingestion',
            'script': 'dm.save_fVar_data() & save_corporate_actions()',
            'description': 'Writes 76 fundamental variables & Alpaca corporate splits/dividends into 1d Zarr array.',
            'frequency': 'Daily at 04:00 AM',
            'status': get_job_status('daily_fundamentals', '04:00 AM', '04:00')
        },
        {
            'time': '04:30 AM',
            'event': 'fundamental_retry',
            'name': 'Fundamental Data Retry Fallback',
            'script': 'dm.save_fVar_data() [Retry]',
            'description': 'Fallback execution run if 04:00 AM fundamental data ingestion was incomplete.',
            'frequency': 'Daily at 04:30 AM',
            'status': get_job_status('fundamental_retry', '04:30 AM', '04:30')
        },
        {
            'time': '09:30 – 16:00',
            'event': 'market_hours_trading',
            'name': 'Market Hours Trading Window',
            'script': 'Market Hours Active Trading Execution',
            'description': 'Monitors market hours trading window across Monday through Friday.',
            'frequency': 'Mon-Fri 09:30 - 16:00',
            'status': get_job_status('market_hours_trading', '09:30 – 16:00', '09:30')
        },
        {
            'time': '5-Min Cadence',
            'event': 'quote_ingestion',
            'name': 'High-Frequency Quote Ingestion',
            'script': 'dm.save_qVar_data(date, time)',
            'description': 'Pulls live level-1 Schwab quote snapshots across u00 universe and writes to 5m Zarr array.',
            'frequency': 'Every 5 Mins (288 Daily Bars)',
            'status': 'Active'
        },
        {
            'time': '23:30 PM',
            'event': 'universe_regeneration',
            'name': 'Universe Regeneration & Day Shell Addition',
            'script': 'UM.regen_csv("u00") & dm.add_db_day_shell()',
            'description': 'Re-scans universe criteria, creates tomorrow day shell, and backfills missing days/splits.',
            'frequency': 'Daily at 23:30 PM',
            'status': get_job_status('universe_regeneration', '23:30 PM', '23:30')
        },
        {
            'time': '23:40 PM',
            'event': 'retention_trim',
            'name': '180-Day Retention Trimming',
            'script': 'dm.retention_trim_db()',
            'description': 'Trims historical data older than 180-day retention window and consolidates Zarr store.',
            'frequency': 'Daily at 23:40 PM',
            'status': get_job_status('retention_trim', '23:40 PM', '23:40')
        },
        {
            'time': '23:45 PM',
            'event': 'monthly_cold_backup',
            'name': 'Monthly Cold Zarr Store Backup',
            'script': 'dm.make_month_cold_backup()',
            'description': 'Archives current and previous month slices to consolidated Zarr stores in data/cold/.',
            'frequency': 'Daily at 23:45 PM',
            'status': get_job_status('monthly_cold_backup', '23:45 PM', '23:45')
        },
        {
            'time': '23:55 PM',
            'event': 'daily_notification',
            'name': 'Daily System Status Notification',
            'script': 'send_daily_notification()',
            'description': 'Dispatches daily pipeline execution report and status summary.',
            'frequency': 'Daily at 23:55 PM',
            'status': get_job_status('daily_notification', '23:55 PM', '23:55')
        }
    ]

    return {
        'jobs': pipeline_jobs,
        'cron_log_text': cron_log_content
    }


def get_symbol_change_logs(months_limit=6):
    """
    Parses universe_change__{MM_YYYY}.log files into structured entries, limited to the last 6 months.
    """
    pattern = os.path.join(LOGS_DIR, 'universe_change__*.log')
    all_files = glob.glob(pattern)
    
    # Sort files by extracted year and month in descending order
    def parse_file_date(filepath):
        basename = os.path.basename(filepath)
        match = re.search(r'universe_change__(\d{2})_(\d{4})\.log', basename)
        if match:
            month, year = int(match.group(1)), int(match.group(2))
            return (year, month)
        return (0, 0)
        
    sorted_files = sorted(all_files, key=parse_file_date, reverse=True)
    recent_files = sorted_files[:months_limit]
    
    entries = []
    
    for filepath in recent_files:
        filename = os.path.basename(filepath)
        try:
            with open(filepath, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
                for line in reversed(lines):
                    match = re.match(r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+-\s+(.*)$', line)
                    if match:
                        timestamp, text = match.group(1), match.group(2)
                        
                        entry_type = 'info'
                        added_count = 0
                        removed_count = 0
                        added_symbols = []
                        removed_symbols = []
                        
                        if text.startswith('Added'):
                            entry_type = 'add'
                            m_add = re.match(r'Added (\d+)\s*(?:stocks|symbols)?:\s*(.*)', text, re.IGNORECASE)
                            if m_add:
                                added_count = int(m_add.group(1))
                                added_symbols = [s.strip() for s in m_add.group(2).split(',') if s.strip()]
                        elif text.startswith('Removed'):
                            entry_type = 'remove'
                            m_rem = re.match(r'Removed (\d+)\s*(?:stocks|symbols)?:\s*(.*)', text, re.IGNORECASE)
                            if m_rem:
                                removed_count = int(m_rem.group(1))
                                removed_symbols = [s.strip() for s in m_rem.group(2).split(',') if s.strip()]
                        elif 'Generated universe' in text:
                            entry_type = 'regen'
                            
                        entries.append({
                            'timestamp': timestamp,
                            'text': text,
                            'type': entry_type,
                            'added_count': added_count,
                            'removed_count': removed_count,
                            'added_symbols': added_symbols,
                            'removed_symbols': removed_symbols,
                            'file': filename
                        })
                    else:
                        entries.append({
                            'timestamp': 'N/A',
                            'text': line,
                            'type': 'info',
                            'added_count': 0,
                            'removed_count': 0,
                            'added_symbols': [],
                            'removed_symbols': [],
                            'file': filename
                        })
        except Exception as e:
            print(f"Error reading log {filename}: {e}")
            
    return entries


def get_system_error_logs():
    """
    Parses symbol error and category error logs from logs/ directory.
    """
    pattern = os.path.join(LOGS_DIR, '*error*.log')
    error_files = sorted(glob.glob(pattern), reverse=True)
    
    error_entries = []
    
    for filepath in error_files:
        filename = os.path.basename(filepath)
        try:
            with open(filepath, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
                for line in reversed(lines):
                    error_entries.append({
                        'filename': filename,
                        'line': line
                    })
        except Exception as e:
            print(f"Error reading error log {filename}: {e}")
            
    return error_entries
