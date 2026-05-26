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

def main():

    st = tm.time()

    datetime_raw = datetime.now()
    datetime_rounded = round_to_nearest_5m(datetime_raw)
    date_str = datetime_rounded.strftime("%Y-%m-%d")
    time_str = datetime_rounded.strftime("%H:%M")
    day_of_week = datetime_rounded.strftime("%A")

    #Always Run:
    print(f"Running Gill Master Script for {date_str} at {time_str}")
    dm = DM()
    dm.save_qVar_data(date_str, time_str)
    #PM.execute_paper_trading(time_str)

    if time_str in return_time_str_range(start='09:30', end='16:00') and day_of_week in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        try:
            PM.execute_paper_trading(time=time_str)
        except:
            print(' Error Running Paper Trading Executions')

    if time_str == '23:40':
        next_day = (datetime_rounded + timedelta(days=1)).strftime("%Y-%m-%d")
        UM.regen_csv('u00')
        dm.add_db_day_shell(next_day)

        if date_str[8:10] == '01':
            dm.make_month_cold_backup(month=date_str[5:7], year=date_str[0:4])

        send_daily_notification()

    if time_str == '04:00':
        dm.save_fVar_data(date_str)

    et = tm.time()
    total_time = round(et - st, 2)

if __name__ == "__main__":
    main()