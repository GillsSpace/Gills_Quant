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

def return_datetime_shift(dt:datetime, mins:int=None, n:int=None) -> datetime:
    """Return a datetime object shifted by a certain number of minutes or 5-minute intervals."""
    if mins is not None:
        shifted_dt = dt + timedelta(minutes=mins)
    elif n is not None:
        shifted_dt = dt + timedelta(minutes=5 * n)
    else:
        shifted_dt = dt
    return shifted_dt

def return_time_str_range(start:str, end:str=None, n=None) -> list:
    """
    Return a list of time strings in 5 minute intervals

    If start and end are provided, return the range between them.
    If start and n are provided, return n intervals starting from start.
    If end and n are provided, return n intervals ending at end.
    """
    time_range = []
    if n is None:
        start_dt = datetime.strptime(start, "%H:%M")
        end_dt = datetime.strptime(end, "%H:%M")
        current_dt = start_dt
        while current_dt <= end_dt:
            time_range.append(current_dt.strftime("%H:%M"))
            current_dt += timedelta(minutes=5)
    elif n is not None:
        if start is not None:
            start_dt = datetime.strptime(start, "%H:%M")
            for i in range(n):
                time_range.append(start_dt.strftime("%H:%M"))
                start_dt += timedelta(minutes=5)
        elif end is not None:
            end_dt = datetime.strptime(end, "%H:%M")
            for i in range(n):
                time_range.append(end_dt.strftime("%H:%M"))
                end_dt -= timedelta(minutes=5)
            time_range.reverse()
    return time_range

def return_day_str_range(start:str, end:str=None, n=None, exclude_weekends:bool=False) -> list:
    """
    Return a list of day strings in daily intervals

    If start and end are provided, return the range between them.
    If start and n are provided, return n intervals starting from start.
    If end and n are provided, return n intervals ending at end.
    """
    day_range = []
    if n is None:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        current_dt = start_dt
        while current_dt <= end_dt:
            if not (exclude_weekends and current_dt.weekday() >= 5):
                day_range.append(current_dt.strftime("%Y-%m-%d"))
            current_dt += timedelta(days=1)
    elif n is not None:
        if start is not None:
            start_dt = datetime.strptime(start, "%Y-%m-%d")
            while len(day_range) < n:
                if not (exclude_weekends and start_dt.weekday() >= 5):
                    day_range.append(start_dt.strftime("%Y-%m-%d"))
                start_dt += timedelta(days=1)
        elif end is not None:
            end_dt = datetime.strptime(end, "%Y-%m-%d")
            while len(day_range) < n:
                if not (exclude_weekends and end_dt.weekday() >= 5):
                    day_range.append(end_dt.strftime("%Y-%m-%d"))
                end_dt -= timedelta(days=1)
            day_range.reverse()
    return day_range

def return_time_str_shift(time:str, mins:int=None, n:int=None) -> str:
    """Return a time string shifted by a certain number of minutes or 5-minute intervals."""
    time_dt = datetime.strptime(time, "%H:%M")
    if mins is not None:
        shifted_dt = time_dt + timedelta(minutes=mins)
    elif n is not None:
        shifted_dt = time_dt + timedelta(minutes=5 * n)
    else:
        shifted_dt = time_dt
    return shifted_dt.strftime("%H:%M")