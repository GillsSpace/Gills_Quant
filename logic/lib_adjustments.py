import os
import json
import numpy as np
import polars as pl
import xarray as xr
from pathlib import Path
from logic.lib_clients import create_client_alpaca_corporate_actions

def get_alpaca_corporate_actions(symbol: str, start_date: str, end_date: str) -> tuple[list, list]:
    """
    Fetches split and dividend corporate actions from Alpaca for a given symbol.
    """
    try:
        from alpaca.data.requests import CorporateActionsRequest
        client = create_client_alpaca_corporate_actions()
        
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
                splits.append({'date': str(ex_date)[:10], 'ratio': ratio})
            elif act_type == 'stock_dividend':
                rate = float(act.get('rate') or 0)
                if rate > 0:
                    splits.append({'date': str(ex_date)[:10], 'ratio': 1.0 + rate})
            elif act_type in ('dividend', 'cash_dividend'):
                amount = float(act.get('amount') or 0)
                dividends.append({'date': str(ex_date)[:10], 'amount': amount})
                
        return splits, dividends
    except Exception as e:
        print(f"Alpaca Corporate Actions failed for {symbol}: {e}")
        return [], []

def extract_db_corporate_actions(zarr_store: xr.Dataset, symbol: str) -> tuple[list, list]:
    """
    Extracts corporate split ratios and dividend amounts directly from the local native Zarr DB store.
    """
    splits = []
    dividends = []
    try:
        ds_1d = zarr_store['1d'].sel(ident=symbol)
        days = [str(d) for d in ds_1d.day.values]
        
        # 1. Read corporate splits if column exists
        if 'corporate.splitRatio' in ds_1d.fVar.values:
            split_ratios = ds_1d.sel(fVar='corporate.splitRatio').values
            for day, ratio in zip(days, split_ratios):
                if not np.isnan(ratio) and ratio > 0 and abs(ratio - 1.0) > 1e-4:
                    splits.append({'date': day, 'ratio': ratio})
                    
        # 2. Read corporate dividend amounts if column exists
        if 'corporate.divAmount' in ds_1d.fVar.values:
            div_amounts = ds_1d.sel(fVar='corporate.divAmount').values
            for day, amount in zip(days, div_amounts):
                if not np.isnan(amount) and amount > 0:
                    dividends.append({'date': day, 'amount': amount})
                    
        # Fallback for dividend ex-dates if corporate.divAmount was empty
        if not dividends and 'fundamental.divExDate' in ds_1d.fVar.values and 'fundamental.divPayAmount' in ds_1d.fVar.values:
            div_ex_dates = ds_1d.sel(fVar='fundamental.divExDate').values
            div_amounts = ds_1d.sel(fVar='fundamental.divPayAmount').values
            seen_dates = set()
            for day, ex_date_num, amount in zip(days, div_ex_dates, div_amounts):
                if np.isnan(ex_date_num) or np.isnan(amount) or amount <= 0:
                    continue
                val = int(ex_date_num)
                year = val // 10000
                month = (val % 10000) // 100
                day_num = val % 100
                if year < 1900 or month < 1 or month > 12 or day_num < 1 or day_num > 31:
                    continue
                ex_date_str = f"{year:04d}-{month:02d}-{day_num:02d}"
                if ex_date_str not in seen_dates:
                    seen_dates.add(ex_date_str)
                    dividends.append({'date': ex_date_str, 'amount': amount})
                    
    except Exception as e:
        print(f"Error extracting DB corporate actions for {symbol}: {e}")
        
    return splits, dividends

def calculate_adjustment_factors(close_prices: pl.DataFrame, splits: list, dividends: list) -> pl.DataFrame:
    """
    Calculates cumulative backward-adjustment factors based on ex-dates of corporate actions.
    Expects close_prices to be a Polars DataFrame with columns ['day', 'close'].
    """
    days = close_prices['day'].to_list()
    closes = close_prices['close'].to_list()
    factors = [1.0] * len(days)
    
    split_map = {s['date']: s['ratio'] for s in splits}
    div_map = {d['date']: d['amount'] for d in dividends}
    
    cum_factor = 1.0
    
    for i in range(len(days) - 1, -1, -1):
        day = days[i]
        factors[i] = cum_factor
        
        # Adjustment applies to all days PRIOR to the ex-dividend / split date.
        if day in split_map:
            ratio = split_map[day]
            if ratio > 0:
                cum_factor /= ratio
                
        if day in div_map:
            div_amount = div_map[day]
            if i > 0:
                prev_close = closes[i - 1]
                if prev_close is not None and not np.isnan(prev_close) and prev_close > 0:
                    mult = 1.0 - div_amount / prev_close
                    if mult > 0:
                        cum_factor *= mult
                    
    return pl.DataFrame({'day': days, 'factor': factors})

def get_adjusted_prices(zarr_store: xr.Dataset, symbol: str, price_var: str = 'quote.mark') -> tuple[pl.Series, pl.Series]:
    """
    Returns the raw and adjusted price series for a symbol as Polars Series,
    calculating adjustment factors directly from the native local Zarr DB store.
    """
    # 1. Load raw prices from 5m array
    da_5m = zarr_store['5m'].sel(ident=symbol, qVar=price_var)
    days = [str(d) for d in da_5m.day.values]
    times = [str(t) for t in da_5m.time.values]
    raw_matrix = da_5m.values  # shape: (num_days, num_times)
    
    day_col = []
    time_col = []
    val_col = []
    for d_idx, d in enumerate(days):
        for t_idx, t in enumerate(times):
            day_col.append(d)
            time_col.append(t)
            val_col.append(raw_matrix[d_idx, t_idx])
            
    raw_df = pl.DataFrame({
        'day': day_col,
        'time': time_col,
        'raw': val_col
    })
    
    # 2. Get daily close prices
    da_1d = zarr_store['1d'].sel(ident=symbol, fVar='quote.closePrice')
    close_days = [str(d) for d in da_1d.day.values]
    close_vals = da_1d.values
    
    valid_mask = ~np.isnan(close_vals)
    close_df = pl.DataFrame({
        'day': [d for d, v in zip(close_days, valid_mask) if v],
        'close': [v for v, m in zip(close_vals, valid_mask) if m]
    })
    
    if close_df.is_empty():
        return raw_df['raw'], raw_df['raw']  # No historical daily data, return raw
        
    # 3. Retrieve corporate actions from native local DB store
    splits, dividends = extract_db_corporate_actions(zarr_store, symbol)
    
    # Fallback to live Alpaca lookup if DB store has no recorded corporate actions
    if not splits and not dividends:
        start_date = close_df['day'][0]
        end_date = close_df['day'][-1]
        splits, dividends = get_alpaca_corporate_actions(symbol, start_date, end_date)
        
    # 4. Calculate adjustment factors
    factors_df = calculate_adjustment_factors(close_df, splits, dividends)
    
    # 5. Broadcast daily factors to 5-minute ticks
    merged_df = raw_df.join(factors_df, on='day', how='left').with_columns(
        pl.col('factor').fill_null(1.0)
    )
    
    adjusted_5m = merged_df['raw'] * merged_df['factor']
    
    return merged_df['raw'], adjusted_5m
