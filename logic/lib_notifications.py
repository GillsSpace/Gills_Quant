import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import python_ntfy


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
    client = python_ntfy.NtfyClient(topic=topic)
    client.send(title=title, message=message)


def send_daily_notification(token_db_path: str | Path | None = None) -> None:
    """Send a daily reminder based on the Schwab refresh-token issuance date."""
    if token_db_path is None:
        token_db_path = Path(__file__).resolve().parent.parent / "secrets" / "tokens.db"

    token_db_path = Path(token_db_path)
    if not token_db_path.exists():
        raise FileNotFoundError(f"Schwab token DB not found: {token_db_path}")

    with sqlite3.connect(token_db_path) as conn:
        row = conn.execute(
            "SELECT refresh_token_issued FROM schwabdev LIMIT 1"
        ).fetchone()

    if row is None or not row[0]:
        raise ValueError("No Schwab refresh token issuance date found in token database")

    issue_datetime = datetime.fromisoformat(row[0]).replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    expiration_period = timedelta(days=7)
    time_left = (issue_datetime + expiration_period - now).total_seconds()

    message = f"Schwab refresh token expires in {int(time_left // 3600)} hours."
    send_ntfy_notification("Daily Schwab Token Check", message)
