import sys
import unittest
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from logic.lib_time import (
    round_to_nearest_5m,
    return_datetime_shift,
    return_time_str_range,
    return_day_str_range,
    return_time_str_shift
)

class TestLibTime(unittest.TestCase):

    def test_round_to_nearest_5m(self):
        dt1 = datetime(2026, 8, 1, 9, 32, 10)
        self.assertEqual(round_to_nearest_5m(dt1), datetime(2026, 8, 1, 9, 30, 0))

        dt2 = datetime(2026, 8, 1, 9, 33, 40)
        self.assertEqual(round_to_nearest_5m(dt2), datetime(2026, 8, 1, 9, 35, 0))

    def test_return_datetime_shift(self):
        dt = datetime(2026, 8, 1, 9, 30)
        shifted_mins = return_datetime_shift(dt, mins=15)
        self.assertEqual(shifted_mins, datetime(2026, 8, 1, 9, 45))

        shifted_n = return_datetime_shift(dt, n=3)
        self.assertEqual(shifted_n, datetime(2026, 8, 1, 9, 45))

    def test_return_time_str_range(self):
        times = return_time_str_range(start="09:30", end="09:45")
        self.assertEqual(times, ["09:30", "09:35", "09:40", "09:45"])

        times_n = return_time_str_range(start="09:30", n=3)
        self.assertEqual(times_n, ["09:30", "09:35", "09:40"])

    def test_return_day_str_range_business_days_count(self):
        """return_day_str_range returns exact requested n business days when excluding weekends."""
        days = return_day_str_range(start="2026-08-07", n=5, exclude_weekends=True)
        self.assertEqual(len(days), 5)
        self.assertEqual(days, ["2026-08-07", "2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13"])

    def test_return_time_str_shift(self):
        shifted = return_time_str_shift("09:30", mins=10)
        self.assertEqual(shifted, "09:40")

        shifted_n = return_time_str_shift("09:30", n=2)
        self.assertEqual(shifted_n, "09:40")


if __name__ == "__main__":
    unittest.main()
