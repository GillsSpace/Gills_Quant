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