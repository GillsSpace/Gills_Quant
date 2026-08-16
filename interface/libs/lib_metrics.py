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
    today_str = days[-1] if days else ""
    # Always recalculate today so live intraday & 04:00 AM fundamental updates are reflected immediately
    missing_days = [d for d in days if d not in cache or d == today_str]
    
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


def get_single_day_metrics_summary(ds, day):
    """
    Returns single day operational metrics for DB Overview cards.
    """
    if ds is None or str(day) not in [str(d) for d in ds.day.values]:
        return {
            'day': str(day),
            'fundamentals_collected': False,
            'times_collected': 0,
            'total_times': 288,
            'active_tickers': 0,
            'total_idents': 0,
            'fill_rate': 0.0,
            'quote_nan_pct': 100.0,
            'edgar_filings': 0
        }
        
    day_str = str(day)
    day_5m = ds['5m'].sel(day=day_str)
    day_1d = ds['1d'].sel(day=day_str)
    
    from datetime import datetime
    now = datetime.now()
    all_days = sorted([str(d) for d in ds.day.values])
    is_today = (day_str == all_days[-1]) or (day_str == now.strftime('%Y-%m-%d'))
    
    if is_today:
        m = now.hour * 60 + now.minute
        expected_times = min(288, (m // 5) + 1)
    else:
        expected_times = 288
    
    # 1. Fundamental data collected
    try:
        close_vals = day_1d.sel(fVar='quote.closePrice').values
        fundamentals_collected = bool(np.any(~np.isnan(close_vals)))
    except Exception:
        fundamentals_collected = False
        
    # 2. 5m time slots collected
    try:
        mark_by_time = (~day_5m.sel(qVar='quote.mark').isnull()).any(dim='ident').values
        times_collected = int(np.sum(mark_by_time))
    except Exception:
        times_collected = 0
    total_times = len(ds.time)
    
    # 3. Active tickers count & Fill Rate vs u00
    try:
        mark_by_ident = (~day_5m.sel(qVar='quote.mark').isnull()).any(dim='time').values
        active_tickers = int(np.sum(mark_by_ident))
    except Exception:
        active_tickers = 0
    total_idents = len(ds.ident)
    fill_rate = round((active_tickers / total_idents) * 100, 2) if total_idents > 0 else 0.0
    
    # 4. General Quote NaN %
    try:
        qvars = [str(q) for q in ds.qVar.values if str(q).startswith('quote.')]
        q_tot = total_times * total_idents * len(qvars)
        q_nan_cnt = int(day_5m.sel(qVar=qvars).isnull().sum().values)
        quote_nan_pct = round((q_nan_cnt / q_tot) * 100, 2) if q_tot > 0 else 0.0
    except Exception:
        quote_nan_pct = 100.0
        
    # 5. EDGAR Filings count (scoped strictly by date slice to avoid bleeding across days)
    edgar_filings = 0
    status_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'status.json')
    
    if is_today:
        todays_filings_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'universes', 'todays_filing_symbols.json')
        if os.path.exists(todays_filings_file):
            try:
                with open(todays_filings_file, 'r') as f:
                    symbols_list = json.load(f)
                    edgar_filings = len(symbols_list)
            except Exception:
                pass
        if edgar_filings == 0 and os.path.exists(status_file):
            try:
                with open(status_file, 'r') as f:
                    s = json.load(f)
                    edgar_filings = s.get('edgar_filings_symbols_today', s.get('edgar_filings_symbols_yesterday', 0))
            except Exception:
                pass
    else:
        # For yesterday's card (all_days[-2]), load edgar_filings_symbols_yesterday from status.json
        yesterday_str = all_days[-2] if len(all_days) > 1 else ""
        if day_str == yesterday_str and os.path.exists(status_file):
            try:
                with open(status_file, 'r') as f:
                    s = json.load(f)
                    edgar_filings = s.get('edgar_filings_symbols_yesterday', 0)
            except Exception:
                pass

    # 6. Fundamental 1d Ingestion Performance & Corporate Actions
    try:
        f_tot = day_1d.size
        f_nan_cnt = int(day_1d.isnull().sum().values)
        f_fill_pct = round(((f_tot - f_nan_cnt) / f_tot) * 100, 2) if f_tot > 0 else 0.0
        f_nan_pct = round((f_nan_cnt / f_tot) * 100, 2) if f_tot > 0 else 100.0
    except Exception:
        f_tot = 0
        f_fill_pct = 0.0
        f_nan_pct = 100.0
        
    div_events = 0
    split_events = 0
    fvars = [str(f) for f in ds.fVar.values]
    if 'corporate.divAmount' in fvars:
        try:
            divs = day_1d.sel(fVar='corporate.divAmount').values
            div_events = int(np.sum(~np.isnan(divs) & (divs > 0)))
        except Exception:
            pass
    if 'corporate.splitRatio' in fvars:
        try:
            splits = day_1d.sel(fVar='corporate.splitRatio').values
            split_events = int(np.sum(~np.isnan(splits) & (splits != 1.0)))
        except Exception:
            pass
            
    return {
        'day': day_str,
        'is_today': is_today,
        'fundamentals_collected': fundamentals_collected,
        'times_collected': times_collected,
        'expected_times': expected_times,
        'total_times': total_times,
        'active_tickers': active_tickers,
        'total_idents': total_idents,
        'fill_rate': fill_rate,
        'quote_nan_pct': quote_nan_pct,
        'edgar_filings': edgar_filings,
        'fund_fill_rate': f_fill_pct,
        'fund_nan_pct': f_nan_pct,
        'fund_total_cells': f_tot,
        'div_events': div_events,
        'split_events': split_events
    }
