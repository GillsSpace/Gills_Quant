import json
from pathlib import Path

import schwabdev as sd
from alpaca.trading.client import TradingClient
from alpaca.data.historical.corporate_actions import CorporateActionsClient


def create_client_schwab():
    """
    Creates and returns a Schwab client using schwabdev library.
    """
    creds_file = Path(__file__).resolve().parent.parent / 'secrets' / 'keys.json'
    tokens_db = Path(__file__).resolve().parent.parent / 'secrets' / 'tokens.db'
    with open(creds_file, 'r') as f:
        keys = json.load(f)

    return sd.Client(
        keys['schwab']['app_key'],
        keys['schwab']['app_secret'],
        tokens_db=str(tokens_db),
    )

def create_client_alpaca_trading(paper: bool = True) -> TradingClient:
    """
    Creates and returns an Alpaca TradingClient using credentials from secrets/keys.json.
    """
    creds_file = Path(__file__).resolve().parent.parent / 'secrets' / 'keys.json'
    with open(creds_file, 'r') as f:
        keys = json.load(f)

    return TradingClient(
        keys['alpaca']['key'],
        keys['alpaca']['secret'],
        paper=paper
    )

def create_client_alpaca_corporate_actions() -> CorporateActionsClient:
    """
    Creates and returns an Alpaca CorporateActionsClient using credentials from secrets/keys.json.
    """
    creds_file = Path(__file__).resolve().parent.parent / 'secrets' / 'keys.json'
    with open(creds_file, 'r') as f:
        keys = json.load(f)

    return CorporateActionsClient(
        keys['alpaca']['key'],
        keys['alpaca']['secret'],
        raw_data=True
    )

def test_client_schwab():
    """
    Tests the Schwab client by fetching and printing account information to standard output.
    """
    client: sd.Client = create_client_schwab()
    accounts = client.account_details_all().json()
    print("Schwab Account Info:")
    for account in accounts:
        print(json.dumps(account, indent=4))

import time
import urllib.request
from datetime import datetime

def ping_schwab_client():
    st = time.time()
    try:
        c = create_client_schwab()
        res = c.account_details_all()
        lat = int((time.time() - st) * 1000)
        return {
            'status': 'ok',
            'code': res.status_code,
            'latency_ms': lat,
            'message': f"200 OK ({lat}ms)"
        }
    except Exception as e:
        return {'status': 'error', 'code': 500, 'latency_ms': 0, 'message': str(e)}

def ping_alpaca_client():
    st = time.time()
    try:
        ac = create_client_alpaca_trading()
        acc = ac.get_account()
        lat = int((time.time() - st) * 1000)
        return {
            'status': 'ok',
            'code': 200,
            'latency_ms': lat,
            'message': f"Active ({lat}ms)"
        }
    except Exception as e:
        return {'status': 'error', 'code': 500, 'latency_ms': 0, 'message': str(e)}

def ping_edgar_client():
    st = time.time()
    try:
        headers = {'User-Agent': 'GillsQuant Research/1.0 (contact@gillsquant.com)'}
        url = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=10-Q&count=10&output=atom'
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            lat = int((time.time() - st) * 1000)
            return {
                'status': 'ok',
                'code': resp.status,
                'latency_ms': lat,
                'message': f"200 OK ({lat}ms)"
            }
    except Exception as e:
        return {'status': 'error', 'code': 500, 'latency_ms': 0, 'message': str(e)}

def ping_all_api_clients():
    return {
        'schwab': ping_schwab_client(),
        'alpaca': ping_alpaca_client(),
        'edgar': ping_edgar_client(),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }