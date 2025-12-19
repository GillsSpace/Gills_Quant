import sys
from pathlib import Path

root_path = Path.cwd().parent if "__file__" not in globals() else Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from logic.UniverseManager import UniverseManager as UM
from logic.DataManager import DataManager as DM
from logic.lib_time import *
from logic.lib_notifications import *

def main():
    datetime_raw = datetime.now()
    datetime_rounded = round_to_nearest_5m(datetime_raw)
    date_str = datetime_rounded.strftime("%Y-%m-%d")
    time_str = datetime_rounded.strftime("%H:%M")

    #Always Run:
    dm = DM()
    dm.save_qVar_data(date_str, time_str)

    if time_str == '23:40':
        next_day = (datetime_rounded + timedelta(days=1)).strftime("%Y-%m-%d")
        UM.regen_csv('u00')
        dm.add_db_day_shell(next_day)
        send_daily_notification()

    if time_str == '04:00':
        dm.save_fVar_data(date_str)

if __name__ == "__main__":
    main()