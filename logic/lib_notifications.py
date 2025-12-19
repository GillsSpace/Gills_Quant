import ntfy
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

def send_ntfy_notification(title: str, message: str) -> None:
    """
    Sends a notification using the ntfy service.

    Parameters
    ----------
    title : str
        The title of the notification.
    message : str
        The body message of the notification.
    """
    topic = "gills_quant_trading"
    ntfy.notify(
        topic=topic,
        title=title,
        message=message,
        priority=3,  # Normal priority
    )

def send_daily_notification():
    token_file = Path(__file__).resolve().parent.parent / 'secrets' / 'tokens.json'
    with open(token_file, 'r') as f:
        tokens = json.load(f)
        issue_date = tokens.get('refresh_token_issued')
        issue_datetime = datetime.fromisoformat(issue_date).replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    expiration_period = timedelta(days=7)
    time_left = (issue_datetime + expiration_period - now).total_seconds()

    message = f"Schwab refresh token expires in {int(time_left // 3600)} hours."

    send_ntfy_notification("Daily Schwab Token Check", message)
