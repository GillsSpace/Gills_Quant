from datetime import datetime, date, time, timedelta
import pandas as pd

def round_to_nearest_5m(dt:datetime) -> datetime:
    """Round a datetime object to the nearest 5 minutes."""
    discard = timedelta(minutes=dt.minute % 5,
                        seconds=dt.second,
                        microseconds=dt.microsecond)
    dt -= discard
    if discard >= timedelta(minutes=2.5):
        dt += timedelta(minutes=5)
    return dt