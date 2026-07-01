import sys
from pathlib import Path
import time as tm

root_path = Path.cwd().parent if "__file__" not in globals() else Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from logic.UniverseManager import UniverseManager as UM
from logic.DataManager import DataManager as DM
from logic.PaperManager import PaperManager as PM
from logic.lib_time import *
from logic.lib_notifications import *

WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

def main():

    st = tm.time()

    datetime_raw = datetime.now()
    datetime_rounded = round_to_nearest_5m(datetime_raw)
    date_str = datetime_rounded.strftime("%Y-%m-%d")
    time_str = datetime_rounded.strftime("%H:%M")
    day_of_week = datetime_rounded.strftime("%A")

    #Always Run:
    print(f"Running Gill Master Script for {date_str} at {time_str}", end="")
    dm = DM()
    dm.save_qVar_data(date_str, time_str)
    #PM.execute_paper_trading(time_str)

    if time_str in return_time_str_range(start='09:30', end='16:00') and day_of_week in WEEKDAYS:
        pass

    if time_str == '23:40':
        next_day = (datetime_rounded + timedelta(days=1)).strftime("%Y-%m-%d")
        UM.regen_csv('u00')
        dm.add_db_day_shell(next_day)
        dm.retention_trim_db()

    if time_str == '23:45':
        month, year = int(date_str[5:7]), int(date_str[0:4])
        dm.make_month_cold_backup(month=month, year=year, overwrite_existing=True)
        prev_month= 12 if month == 1 else (month - 1)
        prev_year = year - 1 if month == 1 else year
        dm.make_month_cold_backup(month=prev_month, year=prev_year, overwrite_existing=True)

    if time_str == '23:55':
        send_daily_notification()

    if time_str == '04:00':
        dm.save_fVar_data(date_str)

    et = tm.time()
    total_time = et - st

    print(f" ({total_time:.2f} seconds)")

if __name__ == "__main__":
    main()