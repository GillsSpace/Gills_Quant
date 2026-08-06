import os
import json
import numpy as np
from logic.DataManager import DataManager

def get_daily_metrics_stats(ds):
    """
    Computes daily stats (close prices, active quote marks, and NaN percentage density)
    and caches them in interface/cache/daily_metrics_cache.json.
    """
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'daily_metrics_cache.json')
    
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cache = json.load(f)
        except Exception as e:
            print(f"Error reading cache: {e}")
            
    days = [str(d) for d in ds.day.values]
    missing_days = [d for d in days if d not in cache]
    
    if missing_days:
        id_len = len(ds.ident)
        t_len = len(ds.time)
        q_len = len(ds.qVar)
        f_len = len(ds.fVar)
        size_5m = t_len * id_len * q_len
        size_1d = id_len * f_len
        tot_size = size_5m + size_1d
        
        qvars = list(ds.qVar.values)
        quote_vars = [v for v in qvars if v.startswith('quote.')]
        ext_vars = [v for v in qvars if v.startswith('extended.')]
        htb_vars = [v for v in qvars if v.startswith('reference.')]
        
        q_tot = t_len * id_len * len(quote_vars)
        ext_tot = t_len * id_len * len(ext_vars)
        htb_tot = t_len * id_len * len(htb_vars)
        
        if len(missing_days) > 10:
            # Batch calculate all days to optimize chunk loading
            close_counts = (~ds['1d'].sel(fVar='quote.closePrice').isnull()).sum(dim='ident').values
            mark_mask = (~ds['5m'].sel(qVar='quote.mark').isnull()).any(dim='time').compute()
            mark_counts = mark_mask.sum(dim='ident').values
            
            is_null_5m = ds['5m'].isnull()
            sum_quote = is_null_5m.sel(qVar=quote_vars).sum(dim=['time', 'ident', 'qVar'])
            sum_extended = is_null_5m.sel(qVar=ext_vars).sum(dim=['time', 'ident', 'qVar'])
            sum_htb = is_null_5m.sel(qVar=htb_vars).sum(dim=['time', 'ident', 'qVar'])
            
            nan_5m = ds['5m'].isnull().sum(dim=['time', 'ident', 'qVar'])
            nan_1d = ds['1d'].isnull().sum(dim=['ident', 'fVar'])
            
            import dask
            q_val, ext_val, htb_val, nan_5m_val, nan_1d_val = dask.compute(
                sum_quote, sum_extended, sum_htb, nan_5m, nan_1d
            )
            
            for i, day in enumerate(days):
                cache[day] = {
                    'close_prices': int(close_counts[i]),
                    'mark_tickers': int(mark_counts[i]),
                    'nan_percent': round(float(((nan_5m_val.values[i] + nan_1d_val.values[i]) / tot_size) * 100), 3),
                    'quote_nan_percent': round(float((q_val.values[i] / q_tot) * 100), 3) if q_tot > 0 else 0.0,
                    'extended_nan_percent': round(float((ext_val.values[i] / ext_tot) * 100), 3) if ext_tot > 0 else 0.0,
                    'htb_nan_percent': round(float((htb_val.values[i] / htb_tot) * 100), 3) if htb_tot > 0 else 0.0
                }
        else:
            # Calculate individually for new days (takes ~0.2s per day)
            for day in missing_days:
                try:
                    day_slice_1d = ds['1d'].sel(day=day)
                    close_count = int((~day_slice_1d.sel(fVar='quote.closePrice').isnull()).sum().values)
                    
                    day_slice_5m = ds['5m'].sel(day=day)
                    mark_count = int((~day_slice_5m.sel(qVar='quote.mark').isnull()).any(dim='time').sum().values)
                    
                    nan_5m_count = int(day_slice_5m.isnull().sum().values)
                    nan_1d_count = int(day_slice_1d.isnull().sum().values)
                    nan_pct = round(((nan_5m_count + nan_1d_count) / tot_size) * 100, 3)
                    
                    q_nan_cnt = int(day_slice_5m.sel(qVar=quote_vars).isnull().sum().values)
                    ext_nan_cnt = int(day_slice_5m.sel(qVar=ext_vars).isnull().sum().values)
                    htb_nan_cnt = int(day_slice_5m.sel(qVar=htb_vars).isnull().sum().values)
                    
                    cache[day] = {
                        'close_prices': close_count,
                        'mark_tickers': mark_count,
                        'nan_percent': nan_pct,
                        'quote_nan_percent': round((q_nan_cnt / q_tot) * 100, 3) if q_tot > 0 else 0.0,
                        'extended_nan_percent': round((ext_nan_cnt / ext_tot) * 100, 3) if ext_tot > 0 else 0.0,
                        'htb_nan_percent': round((htb_nan_cnt / htb_tot) * 100, 3) if htb_tot > 0 else 0.0
                    }
                except Exception as e:
                    print(f"Error calculating stats for {day}: {e}")
                    cache[day] = {
                        'close_prices': 0, 'mark_tickers': 0, 'nan_percent': 100.0,
                        'quote_nan_percent': 100.0, 'extended_nan_percent': 100.0, 'htb_nan_percent': 100.0
                    }
                    
        # Save updated cache
        try:
            with open(cache_path, 'w') as f:
                json.dump(cache, f)
        except Exception as e:
            print(f"Error writing cache: {e}")
            
    # Calculate counts chronologically
    chronological_days = sorted(days)
    close_prices = []
    mark_tickers = []
    nan_percents = []
    quote_nan_percents = []
    extended_nan_percents = []
    htb_nan_percents = []
    
    for day in chronological_days:
        stats = cache.get(day, {
            'close_prices': 0, 'mark_tickers': 0, 'nan_percent': 100.0,
            'quote_nan_percent': 100.0, 'extended_nan_percent': 100.0, 'htb_nan_percent': 100.0
        })
        close_prices.append(stats['close_prices'])
        mark_tickers.append(stats['mark_tickers'])
        nan_percents.append(stats['nan_percent'])
        quote_nan_percents.append(stats.get('quote_nan_percent', 100.0))
        extended_nan_percents.append(stats.get('extended_nan_percent', 100.0))
        htb_nan_percents.append(stats.get('htb_nan_percent', 100.0))
        
    return chronological_days, close_prices, mark_tickers, nan_percents, quote_nan_percents, extended_nan_percents, htb_nan_percents
