import os
import json
import sqlite3
import datetime
import urllib.parse
from pathlib import Path
import requests
import schwabdev as sd

SECRETS_DIR = Path(__file__).resolve().parent.parent.parent / 'secrets'
CREDS_FILE = SECRETS_DIR / 'keys.json'
TOKENS_DB = SECRETS_DIR / 'tokens.db'
UPLOAD_URL = "https://gill-01.taileb5b7d.ts.net/update-token"

def get_keys():
    if not CREDS_FILE.exists():
        raise FileNotFoundError(f"Credentials file not found at {CREDS_FILE}")
    with open(CREDS_FILE, 'r') as f:
        return json.load(f)

def get_token_status():
    """
    Reads tokens.db SQLite database and returns expiration and status details.
    """
    if not TOKENS_DB.exists():
        return {
            "exists": False,
            "message": f"tokens.db file not found at {TOKENS_DB}"
        }

    try:
        conn = sqlite3.connect(f"file:{TOKENS_DB}?mode=ro", uri=True, timeout=30.0)
        cur = conn.cursor()
        cur.execute("SELECT access_token_issued, refresh_token_issued, access_token, refresh_token, expires_in FROM schwabdev LIMIT 1")
        row = cur.fetchone()
        conn.close()

        if not row:
            return {"exists": True, "valid": False, "message": "No tokens stored in database"}

        at_issued_str, rt_issued_str, at_val, rt_val, expires_in = row

        now = datetime.datetime.now(datetime.timezone.utc)
        
        # Parse access token issued time
        try:
            at_issued = datetime.datetime.fromisoformat(at_issued_str)
        except Exception:
            at_issued = None

        # Parse refresh token issued time
        try:
            rt_issued = datetime.datetime.fromisoformat(rt_issued_str)
        except Exception:
            rt_issued = None

        at_valid = False
        at_expires_in_sec = 0
        if at_issued:
            at_age = (now - at_issued).total_seconds()
            at_expires_in_sec = max(0, 1800 - at_age)  # Schwab access token typically 30m (1800s)
            at_valid = at_expires_in_sec > 60

        rt_valid = False
        rt_expires_in_sec = 0
        if rt_issued:
            rt_age = (now - rt_issued).total_seconds()
            rt_expires_in_sec = max(0, 7 * 86400 - rt_age)  # Schwab refresh token valid for 7 days
            rt_valid = rt_expires_in_sec > 3600

        return {
            "exists": True,
            "valid": at_valid and rt_valid,
            "access_token_issued": at_issued_str,
            "refresh_token_issued": rt_issued_str,
            "access_token_valid": at_valid,
            "access_token_expires_in": int(at_expires_in_sec),
            "refresh_token_valid": rt_valid,
            "refresh_token_expires_in": int(rt_expires_in_sec),
            "access_token_preview": f"{at_val[:8]}...{at_val[-4:]}" if at_val and len(at_val) > 12 else "N/A",
            "refresh_token_preview": f"{rt_val[:8]}...{rt_val[-4:]}" if rt_val and len(rt_val) > 12 else "N/A",
        }
    except Exception as e:
        return {"exists": True, "valid": False, "error": str(e)}

def get_auth_url(callback_url: str = "https://127.0.0.1"):
    """
    Generates Schwab OAuth Authorization URL.
    """
    keys = get_keys()
    app_key = keys['schwab']['app_key']
    url = f"https://api.schwabapi.com/v1/oauth/authorize?client_id={app_key}&redirect_uri={urllib.parse.quote(callback_url, safe='')}"
    return {
        "auth_url": url,
        "app_key": app_key,
        "callback_url": callback_url
    }

def upload_tokens_db():
    """
    Uploads tokens.db SQLite database file to Tailscale endpoint.
    """
    if not TOKENS_DB.exists():
        return {"success": False, "message": f"tokens.db does not exist at {TOKENS_DB}"}

    try:
        with open(TOKENS_DB, 'rb') as f:
            files = {'file': f}
            response = requests.post(UPLOAD_URL, files=files, timeout=15)
            return {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response": response.json() if response.headers.get("content-type") == "application/json" else response.text
            }
    except Exception as e:
        return {"success": False, "error": str(e)}

def perform_token_update(redirect_url_or_code: str = None):
    """
    Performs token update using schwabdev client.
    If redirect_url_or_code is supplied, completes full OAuth code flow.
    Otherwise attempts force token update and uploads to remote server.
    """
    keys = get_keys()
    app_key = keys['schwab']['app_key']
    app_secret = keys['schwab']['app_secret']

    if redirect_url_or_code:
        # Create client with custom call_on_auth to pass the provided redirect code
        def custom_auth(auth_url):
            return redirect_url_or_code

        client = sd.Client(
            app_key,
            app_secret,
            tokens_db=str(TOKENS_DB),
            call_on_auth=custom_auth,
            open_browser_for_auth=False
        )
        updated = client.tokens.update_tokens(force_refresh_token=True)
    else:
        client = sd.Client(
            app_key,
            app_secret,
            tokens_db=str(TOKENS_DB),
            open_browser_for_auth=False
        )
        updated = client.tokens.update_tokens(force_access_token=True, force_refresh_token=True)

    upload_result = upload_tokens_db()
    status = get_token_status()

    return {
        "updated": updated,
        "upload_result": upload_result,
        "token_status": status
    }
