import os
import zarr
import json
import shutil
import warnings
import time as tm
import numpy as np
import xarray as xr

from pathlib import Path
from datetime import datetime, timedelta, time, date

from logic.lib_time import *
from logic.UniverseManager import UniverseManager as UM
from logic.DataManager import DataManager as DM

from alpaca.trading.client import TradingClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.corporate_actions import CorporateActionsClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.stream import TradingStream
from alpaca.data.live.stock import StockDataStream

from alpaca.data.requests import (
    CorporateActionsRequest,
    StockBarsRequest,
    StockQuotesRequest,
    StockTradesRequest,
)
from alpaca.trading.requests import (
    ClosePositionRequest,
    GetAssetsRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopLossRequest,
    StopOrderRequest,
    TakeProfitRequest,
    TrailingStopOrderRequest,
)
from alpaca.trading.enums import (
    AssetExchange,
    AssetStatus,
    OrderClass,
    OrderSide,
    OrderType,
    QueryOrderStatus,
    TimeInForce,
)

warnings.filterwarnings("ignore", message=".*Zarr format 3.*")

class PaperManager:

    @staticmethod
    def _create_alpaca_client():
        """
        Creates and returns an Alpaca client using the alpaca-trade-api library.
        """
        creds_file = Path(__file__).resolve().parent.parent / 'secrets' / 'keys.json'
        with open(creds_file, 'r') as f:
            keys = json.load(f)

        alpaca_key = keys['alpaca']['key']
        alpaca_secret = keys['alpaca']['secret']

        return TradingClient(alpaca_key, alpaca_secret, paper=True)
    
    @staticmethod
    def execute_paper_trading(time):
        with DM.return_hot_store() as zarr_store:
            if zarr_store is None:
                print("No hot store found.")
                return
            client = PaperManager._create_alpaca_client()

            try:
                data = client.get_open_position('AAPL')
                position = float(data.qty)
            except:
                position = 0.0

            trade_in = True
            trade_out = True
            trade_halt = False

            old_price = zarr_store['5m'].sel(ident='AAPL',qVar='quote.mark',time=return_time_str_shift(time,n=-3)).values[-1]
            print(f"    Price at {return_time_str_shift(time,n=-3)}: {old_price}")

            for i in range(3):
                price = zarr_store['5m'].sel(ident='AAPL',qVar='quote.mark',time=return_time_str_shift(time,n=(-2+i))).values[-1]
                print(f"    Price at {return_time_str_shift(time,n=(-2+i))}: {price}")
                if price > old_price:
                    trade_in = False
                if price < old_price:
                    trade_out = False
                if np.isnan(price):
                    trade_halt = True
                old_price = price
            
            if trade_in and not trade_halt:
                if position < 200:
                    order = PaperManager._gen_order('AAPL','buy',qty=2)
                    client.submit_order(order)
                    print('    Order Executed: Buy Trade Placed')

            if trade_out and not trade_halt:
                if position > 1:
                    order = PaperManager._gen_order('AAPL','sell',qty=2)
                    client.submit_order(order)
                    print('    Order Executed: Sell Trade Placed')

    @staticmethod
    def _gen_order(ident,side,notional=None,qty=None):
        side = OrderSide.SELL if side in ['sell','s'] else OrderSide.BUY
        if notional:
            return MarketOrderRequest(
                symbol = ident,
                notional = notional,
                side = side,
                type = OrderType.MARKET,
                time_in_force = TimeInForce.DAY,
            )
        elif qty:
            return MarketOrderRequest(
                symbol = ident,
                qty = qty,
                side = side,
                type = OrderType.MARKET,
                time_in_force = TimeInForce.DAY,
            )
        else:
            raise Exception('Must provide noional amount or quantity')

        