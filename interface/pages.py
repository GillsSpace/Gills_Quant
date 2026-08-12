from flask import Flask, Blueprint, render_template
from datetime import datetime, timedelta, timezone
import os

from logic.lib_clients import *
from logic.DataManager import DataManager
from interface.libs.lib_dash import get_dashboard_stats

bp = Blueprint('main', __name__)

@bp.route('/')
def home():
    sd_client = create_client_schwab()
    rt_delta = timedelta(seconds=sd_client.tokens._refresh_token_timeout) - (datetime.now(timezone.utc) - sd_client.tokens._refresh_token_issued)
    
    # Format the refresh token expiration countdown nicely with days, hours, and minutes
    days = rt_delta.days
    hours, remainder = divmod(rt_delta.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    
    if days > 0:
        token_str = f"Refresh token expires in: {days}d {hours}h {minutes}m"
    else:
        token_str = f"Refresh token expires in: {hours}h {minutes}m"
    
    stats = get_dashboard_stats()
    
    return render_template('pages/home.html', tokens=token_str, stats=stats)
from interface.libs.lib_metrics import get_daily_metrics_stats


@bp.route('/database')
def database():
    ds = DataManager.return_hot_store()
    
    if ds is not None:
        chronological_days, close_prices, mark_tickers, nan_percents, quote_nans, extended_nans, htb_nans = get_daily_metrics_stats(ds)
        
        # Build mapping from day to stats, accounting for inactive vs missed NaNs
        day_stats = {}
        total_tickers = len(ds.ident)
        for i, d in enumerate(chronological_days):
            inactive_pct = round(((total_tickers - mark_tickers[i]) / total_tickers) * 100, 3) if total_tickers > 0 else 0.0
            missed_pct = round(max(0.0, nan_percents[i] - inactive_pct), 3)
            day_stats[d] = {
                'close_prices': close_prices[i],
                'mark_tickers': mark_tickers[i],
                'nan_percent': nan_percents[i],
                'inactive_nan_percent': inactive_pct,
                'missed_nan_percent': missed_pct,
                'quote_nan_percent': quote_nans[i],
                'extended_nan_percent': extended_nans[i],
                'htb_nan_percent': htb_nans[i]
            }
            
        # Reverse chronological for table
        table_data = []
        for d in reversed(chronological_days):
            table_data.append({
                'day': d,
                'close_prices': day_stats[d]['close_prices'],
                'mark_tickers': day_stats[d]['mark_tickers'],
                'nan_percent': day_stats[d]['nan_percent'],
                'inactive_nan_percent': day_stats[d]['inactive_nan_percent'],
                'missed_nan_percent': day_stats[d]['missed_nan_percent'],
                'quote_nan_percent': day_stats[d]['quote_nan_percent'],
                'extended_nan_percent': day_stats[d]['extended_nan_percent'],
                'htb_nan_percent': day_stats[d]['htb_nan_percent']
            })
            
        db_info = {
            'days': sorted(chronological_days, reverse=True),
            'days_json': json.dumps(chronological_days),
            'close_prices_json': json.dumps(close_prices),
            'mark_tickers_json': json.dumps(mark_tickers),
            'nan_percents_json': json.dumps(nan_percents),
            'quote_nan_percents_json': json.dumps(quote_nans),
            'extended_nan_percents_json': json.dumps(extended_nans),
            'htb_nan_percents_json': json.dumps(htb_nans),
            'inactive_nan_percents_json': json.dumps([day_stats[d]['inactive_nan_percent'] for d in chronological_days]),
            'missed_nan_percents_json': json.dumps([day_stats[d]['missed_nan_percent'] for d in chronological_days]),
            'table_data': table_data,
            'idents': sorted([str(i) for i in ds.ident.values]),
            'qvars': sorted([str(q) for q in ds.qVar.values]),
            'fvars': sorted([str(f) for f in ds.fVar.values]),
            'num_days': len(ds.day),
            'num_idents': len(ds.ident),
            'num_qvars': len(ds.qVar),
            'num_fvars': len(ds.fVar),
            'path': str(DataManager.hot_path_db),
            'retention': DataManager.hot_data_retention_days
        }
        
        # Calculate database size in GB
        total_size = 0
        db_path = DataManager.hot_path_db
        if os.path.exists(db_path):
            for dirpath, dirnames, filenames in os.walk(db_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total_size += os.path.getsize(fp)
        db_info['db_size_gb'] = round(total_size / (1024 * 1024 * 1024), 3)
    else:
        db_info = None
        
    return render_template('pages/database.html', db_info=db_info)


@bp.route('/api/ticker/<symbol>')
def get_ticker_stats(symbol):
    import numpy as np
    import json
    
    ds = DataManager.return_hot_store()
    if ds is None:
        return {"error": "Database not loaded"}, 400
        
    try:
        symbol_str = str(symbol).upper().strip()
        if symbol_str not in ds.ident.values:
            return {"error": f"Symbol {symbol_str} not found in database"}, 404
            
        sym_1d = ds['1d'].sel(ident=symbol_str)
        close_prices = sym_1d.sel(fVar='quote.closePrice').values
        days = [str(d) for d in ds.day.values]
        
        chart_data = []
        for d, price in zip(days, close_prices):
            if not np.isnan(price):
                chart_data.append((d, float(price)))
                
        chart_dates = [x[0] for x in chart_data]
        chart_prices = [x[1] for x in chart_data]
        
        first_date = chart_dates[0] if chart_dates else "N/A"
        last_date = chart_dates[-1] if chart_dates else "N/A"
        total_days_active = len(chart_dates)
        
        stats = {
            "symbol": symbol_str,
            "exchange": "N/A",
            "asset_type": "N/A",
            "latest_close": "N/A",
            "pe_ratio": "N/A",
            "eps": "N/A",
            "div_yield": "N/A",
            "div_freq": "N/A",
            "volume_10d": "N/A",
            "volume_1yr": "N/A",
            "first_date": first_date,
            "last_date": last_date,
            "total_days": total_days_active,
            "chart_dates": chart_dates,
            "chart_prices": chart_prices
        }
        
        if last_date != "N/A":
            latest_slice = sym_1d.sel(day=last_date)
            
            def get_val(fvar, is_float=False):
                try:
                    val = latest_slice.sel(fVar=fvar).values
                    if hasattr(val, 'item'):
                        val = val.item()
                    if val is None or str(val) in ['nan', 'NaN', 'None', '']:
                        return "N/A"
                    if isinstance(val, (int, float)) and np.isnan(val):
                        return "N/A"
                    if is_float:
                        return float(val)
                    return str(val)
                except Exception:
                    return "N/A"
            
            # Map Categorical Exchange Code
            exchange_idx = get_val("reference.exchange", is_float=True)
            exchange_map = {0.0: "NYSE", 1.0: "AMEX", 2.0: "Exchange 9", 3.0: "ARCA", 4.0: "NASDAQ"}
            stats["exchange"] = exchange_map.get(exchange_idx, "N/A") if isinstance(exchange_idx, float) else "N/A"
            
            # Map Categorical Asset Subtype Code
            subtype_idx = get_val("assetSubType", is_float=True)
            subtype_map = {
                0.0: "ADR (American Depositary Receipt)",
                1.0: "Common Stock",
                2.0: "Preferred Stock",
                3.0: "Unit Investment Trust (UIT)",
                4.0: "Closed-End Fund (CEF)"
            }
            stats["asset_type"] = subtype_map.get(subtype_idx, "N/A") if isinstance(subtype_idx, float) else "N/A"
            
            latest_close = get_val("quote.closePrice", is_float=True)
            stats["latest_close"] = round(latest_close, 2) if isinstance(latest_close, float) else "N/A"
            
            pe = get_val("fundamental.peRatio", is_float=True)
            stats["pe_ratio"] = round(pe, 2) if isinstance(pe, float) else "N/A"
            
            eps = get_val("fundamental.eps", is_float=True)
            stats["eps"] = round(eps, 2) if isinstance(eps, float) else "N/A"
            
            div_yield = get_val("fundamental.divYield", is_float=True)
            stats["div_yield"] = f"{round(div_yield, 2)}%" if isinstance(div_yield, float) else "N/A"
            
            div_freq = get_val("fundamental.divFreq", is_float=True)
            freq_map = {1.0: "Annual", 2.0: "Semi-Annual", 4.0: "Quarterly", 12.0: "Monthly"}
            stats["div_freq"] = freq_map.get(div_freq, f"Code {int(div_freq)}") if isinstance(div_freq, float) else "N/A"
            
            vol_10d = get_val("fundamental.avg10DaysVolume", is_float=True)
            stats["volume_10d"] = f"{round(vol_10d / 1e6, 2)}M" if isinstance(vol_10d, float) else "N/A"
            
            vol_1yr = get_val("fundamental.avg1YearVolume", is_float=True)
            stats["volume_1yr"] = f"{round(vol_1yr / 1e6, 2)}M" if isinstance(vol_1yr, float) else "N/A"
            
        return stats
    except Exception as e:
        print(f"Error fetching stats for {symbol}: {e}")
        return {"error": str(e)}, 500

@bp.route('/settings')
def settings():
    return render_template('pages/settings.html')




