import os
import zarr
import json
import shutil
import warnings
import time as tm
import numpy as np
import xarray as xr
import pandas as pd

from pathlib import Path
from pandas.api.types import CategoricalDtype
from datetime import datetime, timedelta, time, date

from logic.lib_time import *
from logic.UniverseManager import UniverseManager as UM
from logic.DataManager import DataManager as DM

warnings.filterwarnings("ignore", message=".*Zarr format 3.*")

class BaseStrategy:
    def __init__(self, universe_id="u00", allocation_dollars=2000):
        self.universe_id = universe_id
        self.allocation_dollars = allocation_dollars
        self.dm = DM()
        self.um = UM()
        self.universe = self.um.return_universe_list(self.universe_id)

    def predict(self, current_time):
        raise NotImplementedError("Subclasses must implement this method.")
    
class SimpleMeanReversionStrategy(BaseStrategy):
    def __init__(self, universe_id="u00", lookback_period=20, entry_threshold=1.5, exit_threshold=0.5):
        super().__init__(universe_id)
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def predict(self, current_time):
        signals = {}

        zarr_store = self.dm.return_hot_store()
        start_date = (current_time - timedelta(days=self.lookback_period)).strftime("%Y-%m-%d")
        days = return_day_str_range(start_date, current_time.strftime("%Y-%m-%d"), exclude_weekends=True)
        times = return_time_str_range(start='09:30', end='16:00')
        historical_data = zarr_store['5m'].sel(day=days, time=times)

        if historical_data is None or len(historical_data.day.values) < self.lookback_period:
            raise ValueError(f"Not enough historical data for lookback period of {self.lookback_period} days.")

        for ticker in self.universe:
            try:
                ticker_data = historical_data.sel(ident=ticker)
                # Select the 'quote.mark' field for prices
                ticker_prices = ticker_data.sel(qVar='quote.mark')
                flat_prices = ticker_prices.values.ravel()
                # Filter out NaNs (e.g. from weekends or after-hours when market is closed)
                flat_prices = flat_prices[~np.isnan(flat_prices)]
                
                if len(flat_prices) == 0:
                    continue

                # Calculate mean and standard deviation
                mean_price = np.mean(flat_prices)
                std_price = np.std(flat_prices)
                current_price = flat_prices[-1]

                # Generate signals based on thresholds
                if current_price > mean_price + self.entry_threshold * std_price:
                    signals[ticker] = 'sell'
                elif current_price < mean_price - self.entry_threshold * std_price:
                    signals[ticker] = 'buy'
                elif abs(current_price - mean_price) < self.exit_threshold * std_price:
                    signals[ticker] = 'hold'
            except Exception as e:
                continue
        
        return signals

    def return_optimal_position(self):
        signals = self.predict(datetime.now())
        num_signals = sum(1 for signal in signals.values() if signal in ['buy', 'sell'])
        if num_signals == 0:
            return {ticker: 0 for ticker in self.universe}

        allocation_per_signal = self.allocation_dollars / num_signals
        position_sizes = {}
        for ticker, signal in signals.items():
            if signal == 'buy':
                position_sizes[ticker] = allocation_per_signal
            elif signal == 'sell':
                position_sizes[ticker] = -allocation_per_signal
            else:
                position_sizes[ticker] = 0
        
        return position_sizes








