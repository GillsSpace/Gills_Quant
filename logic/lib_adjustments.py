import os
import json
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# Try importing Alpaca if available
try:
    from alpaca.data.historical.corporate_actions import CorporateActionsClient
    from alpaca.data.requests import CorporateActionsRequest
    HAS_ALPACA = True
except ImportError:
    HAS_ALPACA = False

def get_alpaca_corporate_actions(symbol: str, start_date: str, end_date: str) -> tuple[list, list]:
    """
    Fetches split and dividend corporate actions from Alpaca for a given symbol.
    """
    if not HAS_ALPACA:
        raise ImportError("Alpaca Python SDK is not installed. Please install it using 'pip install alpaca-py'.")

    creds_file = Path(__file__).resolve().parent.parent / 'secrets' / 'keys.json'
    if not creds_file.exists():
        raise FileNotFoundError("secrets/keys.json not found.")
        
    with open(creds_file, 'r') as f:
        keys = json.load(f)
        
    alpaca_key = keys['alpaca']['key']
    alpaca_secret = keys['alpaca']['secret']
    
    client = CorporateActionsClient(alpaca_key, alpaca_secret, raw_data=True)
    
    request = CorporateActionsRequest(
        symbols=[symbol],
        start=start_date,
        end=end_date
    )
    
    raw_data = client.get_corporate_actions(request)
    
    splits = []
    dividends = []
    
    actions = []
    if isinstance(raw_data, dict):
        for k, v in raw_data.items():
            if isinstance(v, list):
                actions.extend(v)
    elif isinstance(raw_data, list):
        actions = raw_data
        
    for act in actions:
        ex_date = act.get('ex_date')
        if not ex_date:
            continue
        
        act_type = act.get('action_type')
        if act_type in ('split', 'forward_split', 'reverse_split'):
            new_rate = float(act.get('new_rate') or 1)
            old_rate = float(act.get('old_rate') or 1)
            ratio = new_rate / old_rate if old_rate != 0 else 1.0
            splits.append({'date': ex_date, 'ratio': ratio})
        elif act_type == 'stock_dividend':
            rate = float(act.get('rate') or 0)
            if rate > 0:
                splits.append({'date': ex_date, 'ratio': 1.0 + rate})
        elif act_type in ('dividend', 'cash_dividend'):
            amount = float(act.get('amount') or 0)
            dividends.append({'date': ex_date, 'amount': amount})
            
    return splits, dividends

def extract_db_dividends(zarr_store: xr.Dataset, symbol: str) -> list:
    """
    Extracts dividend ex-dates and amounts from local daily database metadata.
    """
    try:
        ds_1d = zarr_store['1d'].sel(ident=symbol)
        div_ex_dates = ds_1d.sel(fVar='fundamental.divExDate').to_pandas()
        div_amounts = ds_1d.sel(fVar='fundamental.divPayAmount').to_pandas()
        
        valid_mask = div_ex_dates.notna() & div_amounts.notna()
        div_ex_dates = div_ex_dates[valid_mask]
        div_amounts = div_amounts[valid_mask]
        
        dividends = []
        seen_dates = set()
        for day, ex_date_num in div_ex_dates.items():
            val = int(ex_date_num)
            year = val // 10000
            month = (val % 10000) // 100
            day_num = val % 100
            if year < 1900 or month < 1 or month > 12 or day_num < 1 or day_num > 31:
                continue
            ex_date_str = f"{year:04d}-{month:02d}-{day_num:02d}"
            
            if ex_date_str not in seen_dates:
                seen_dates.add(ex_date_str)
                amount = div_amounts.loc[day]
                if amount > 0:
                    dividends.append({'date': ex_date_str, 'amount': amount})
        return dividends
    except Exception as e:
        print(f"Error extracting DB dividends for {symbol}: {e}")
        return []

def calculate_adjustment_factors(close_prices: pd.Series, splits: list, dividends: list) -> pd.Series:
    """
    Calculates cumulative backward-adjustment factors based on ex-dates of corporate actions.
    """
    factors = pd.Series(1.0, index=close_prices.index)
    
    split_map = {s['date']: s['ratio'] for s in splits}
    div_map = {d['date']: d['amount'] for d in dividends}
    
    cum_factor = 1.0
    days = close_prices.index.tolist()
    
    for i in range(len(days) - 1, -1, -1):
        day = days[i]
        factors.iloc[i] = cum_factor
        
        # Adjustment applies to all days PRIOR to the ex-dividend / split date.
        # Check if today is the ex-date, update the cumulative multiplier for next iteration.
        if day in split_map:
            ratio = split_map[day]
            if ratio > 0:
                cum_factor /= ratio
                
        if day in div_map:
            div_amount = div_map[day]
            if i > 0:
                prev_close = close_prices.iloc[i - 1]
                if not np.isnan(prev_close) and prev_close > 0:
                    mult = 1.0 - div_amount / prev_close
                    if mult > 0:
                        cum_factor *= mult
                    
    return factors

def get_adjusted_prices(zarr_store: xr.Dataset, symbol: str, price_var: str = 'quote.mark', use_alpaca: bool = False) -> tuple[pd.Series, pd.Series]:
    """
    Returns the raw and adjusted price series for a symbol.
    
    Returns:
        (raw_prices, adjusted_prices) as pandas Series.
    """
    # 1. Load raw prices
    raw_5m = zarr_store['5m'].sel(ident=symbol, qVar=price_var).stack(timeline=('day', 'time')).to_pandas()
    # Normalize index to timezone-naive datetimes
    raw_5m.index = pd.to_datetime([f"{d} {t}" for d, t in raw_5m.index])
    
    # 2. Get daily close prices
    close_prices = zarr_store['1d'].sel(ident=symbol, fVar='quote.closePrice').to_pandas().dropna()
    if close_prices.empty:
        return raw_5m, raw_5m  # No historical daily data, return raw
        
    # 3. Retrieve corporate actions
    splits = []
    dividends = []
    if use_alpaca and HAS_ALPACA:
        start_date = close_prices.index[0]
        end_date = close_prices.index[-1]
        try:
            splits, dividends = get_alpaca_corporate_actions(symbol, start_date, end_date)
        except Exception as e:
            print(f"Alpaca Corporate Actions failed for {symbol}: {e}. Falling back to DB dividends only.")
            dividends = extract_db_dividends(zarr_store, symbol)
    else:
        dividends = extract_db_dividends(zarr_store, symbol)
        
    # 4. Calculate adjustment factors
    daily_factors = calculate_adjustment_factors(close_prices, splits, dividends)
    
    # 5. Broadcast daily factors to 5-minute ticks
    # Create a mapping from day string 'YYYY-MM-DD' to factor
    factor_map = daily_factors.to_dict()
    
    # Map index dates back to the factors
    tick_days = raw_5m.index.strftime('%Y-%m-%d')
    tick_factors = tick_days.map(factor_map).fillna(1.0)
    
    # Apply adjustments
    adjusted_5m = raw_5m * tick_factors
    
    return raw_5m, adjusted_5m
