import sys
from pathlib import Path
import time as tm
from datetime import datetime, timedelta

root_path = Path.cwd().parent if "__file__" not in globals() else Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from logic.UniverseManager import UniverseManager as UM
from logic.DataManager import DataManager as DM
from logic.lib_time import round_to_nearest_5m
from logic.lib_notifications import send_daily_notification
from logic.lib_edgar import update_ticker_cik_map, detect_todays_filing_symbols, update_current_edgar_data_file
from logic.lib_files import reset_daily_status, update_status, update_cron_status

WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

def main():

    st = tm.time()

    datetime_raw = datetime.now()
    datetime_rounded = round_to_nearest_5m(datetime_raw)
    date_str = datetime_rounded.strftime("%Y-%m-%d")
    time_str = datetime_rounded.strftime("%H:%M")
    day_of_week = datetime_rounded.strftime("%A")

    extra_print_times = time_str in ['00:00', '03:15', '03:20', '03:25', '04:00', '04:30', '23:30', '23:40', '23:45', '23:55']

    # Always Run: 5-Minute Quote Ingestion ===============================================

    if extra_print_times:
        print(f"Running Gill Master Script for {date_str} at {time_str}", flush=True)
    else:
        print(f"Running Gill Master Script for {date_str} at {time_str}", end="", flush=True)

    dm = DM()
    dm.save_qVar_data(date_str, time_str)
    update_status({"last_qvar_pull_time": f"{date_str} {time_str}"})

    # ===================================================================================

    # Scheduled Tasks: ==================================================================

    if time_str == '00:00':
        reset_daily_status()
        print("\t[00:00 Daily Reset] Reset operational statuses and cycled EDGAR filings", flush=True)
        update_cron_status('daily_status_reset', 'Completed')

    elif time_str == '03:15':
        ticker_map = update_ticker_cik_map(max_retries=5)
        print(f"\tUpdated SEC Ticker->CIK map ({len(ticker_map) if ticker_map else 0} tickers)", flush=True)
        update_cron_status('sec_ticker_cik_map', 'Completed')

    elif time_str == '03:20':
        filing_symbols = detect_todays_filing_symbols(max_retries=5)
        filings_count = len(filing_symbols) if filing_symbols else 0
        print(f"\tDetected {filings_count} universe filings for today", flush=True)
        update_status({
            "edgar_filings_symbols_today": filings_count,
            "cron_status": {"sec_rss_filings": "Completed"}
        })

    elif time_str == '03:25':
        edgar_df = update_current_edgar_data_file(max_retries=5)
        total_syms = len(edgar_df) if edgar_df is not None and hasattr(edgar_df, '__len__') else 0
        print(f"\tUpdated current_edgar_data.parquet (Total Symbols: {total_syms})", flush=True)
        update_cron_status('sec_xbrl_facts', 'Completed')

    elif time_str == '04:00':
        dm.save_fVar_data(date_str)
        dm.save_corporate_actions_for_day(date_str)
        update_status({
            "fundamental_data_pulled_today": True,
            "cron_status": {"daily_fundamentals": "Completed"}
        })

    elif time_str == '04:30':
        if not dm.has_fundamental_data(date_str):
            print(f"\n\tRetrying fundamental data fetch for {date_str} at 04:30...", flush=True)
            dm.save_fVar_data(date_str)
            dm.save_corporate_actions_for_day(date_str)
            update_status({
                "fundamental_data_pulled_today": True,
                "cron_status": {"fundamental_retry": "Completed"}
            })
        else:
            update_cron_status('fundamental_retry', 'Completed')

    elif time_str == '23:30':
        next_day = (datetime_rounded + timedelta(days=1)).strftime("%Y-%m-%d")
        u_stats = UM.regen_csv('u00')
        dm.add_db_day_shell(next_day)
        dm.backfill_missing_days_and_corporate_actions()
        update_status({
            "u00_symbols_count": u_stats.get('symbols_count', u_stats.get('new_size', 0)),
            "u00_asset_breakdown": u_stats.get('asset_breakdown', {}),
            "cron_status": {"universe_regeneration": "Completed"}
        })

    elif time_str == '23:40':
        trim_stats = dm.retention_trim_db()
        active_days = trim_stats.get('num_days_after', 0) if trim_stats else 0
        update_status({
            "hot_db_active_days_count": active_days,
            "hot_db_disk_size_mb": dm.get_hot_db_disk_size_mb(),
            "cron_status": {"retention_trim": "Completed"}
        })

    elif time_str == '23:45':
        month, year = int(date_str[5:7]), int(date_str[0:4])
        dm.make_month_cold_backup(month=month, year=year, overwrite_existing=True)
        prev_month = 12 if month == 1 else (month - 1)
        prev_year = year - 1 if month == 1 else year
        dm.make_month_cold_backup(month=prev_month, year=prev_year, overwrite_existing=True)
        update_cron_status('monthly_cold_backup', 'Completed')

    elif time_str == '23:55':
        send_daily_notification()
        update_cron_status('daily_notification', 'Completed')

    # ===================================================================================

    et = tm.time()
    total_time = et - st

    if extra_print_times:
        print(f"\tCompleted in {total_time:.2f} seconds", flush=True)
    else:
        print(f" ({total_time:.2f} seconds)", flush=True)

if __name__ == "__main__":
    main()