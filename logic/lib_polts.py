#Imports
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime

def plot_zarr_day(zarr_store, day, qVar='quote.mark', ident='AAPL', connect_gaps=False):
    """Extracts 5m data and plots it with a cleaned-up x-axis for one or more idents, handling gaps."""
    idents = [ident] if isinstance(ident, str) else ident
    
    plt.figure(figsize=(12, 5))
    
    for i in idents:
        # Extract data (contains NaNs for missing intervals)
        da_5m = zarr_store['5m'].sel(day=day, qVar=qVar, ident=i)
        times = [str(t) for t in da_5m.time.values]
        vals = da_5m.values
        
        # Combine day and time into valid datetimes for correct gap rendering
        x_dates = [datetime.strptime(f"{day} {t}", "%Y-%m-%d %H:%M") for t in times]
        
        # Plot main line with gaps
        line = plt.plot(x_dates, vals, label=i, linewidth=1.5)
        
        # Conditionally bridge gaps with a transparent line
        if connect_gaps:
            valid = [(x, v) for x, v in zip(x_dates, vals) if not np.isnan(v)]
            if valid:
                x_clean, v_clean = zip(*valid)
                plt.plot(x_clean, v_clean, color=line[0].get_color(), alpha=0.3, linewidth=1.5)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(12)) 
    plt.xticks(rotation=45, ha='right')
    
    plt.title(f"{', '.join(idents)}: {qVar} ({day})", loc='left', fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_zarr_range(zarr_store, start_day, end_day, qVar='', fVar='', ident='AAPL', connect_gaps=False):
    """Plots a range of days for one or more idents, leaving gaps for missing dates or bridging them transparently."""
    idents = [ident] if isinstance(ident, str) else ident

    if (qVar == '' and fVar == '') or qVar != '':
        if qVar == '':
            qVar = 'quote.mark'

        plt.figure(figsize=(14, 6))
        
        for i in idents:
            da_5m = zarr_store['5m'].sel(
                day=slice(start_day, end_day), 
                qVar=qVar, 
                ident=i
            )
            days = [str(d) for d in da_5m.day.values]
            times = [str(t) for t in da_5m.time.values]
            matrix = da_5m.values
            
            x_dates = [datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M") for d in days for t in times]
            vals = matrix.ravel()
            
            # Plot main line with NaNs (creates gaps)
            line = plt.plot(x_dates, vals, label=i, linewidth=1)
            
            # Conditionally plot continuous line to bridge gaps
            if connect_gaps:
                valid = [(x, v) for x, v in zip(x_dates, vals) if not np.isnan(v)]
                if valid:
                    x_clean, v_clean = zip(*valid)
                    plt.plot(x_clean, v_clean, color=line[0].get_color(), alpha=0.3, linewidth=1)
        
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
        
        plt.title(f"{', '.join(idents)}: {qVar} ({start_day} to {end_day})", loc='left', fontsize=14)
        plt.xticks(rotation=30, ha='right')
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    elif fVar != '':
        plt.figure(figsize=(14, 6))
        
        for i in idents:
            da_1d = zarr_store['1d'].sel(
                day=slice(start_day, end_day), 
                fVar=fVar, 
                ident=i
            )
            days = [str(d) for d in da_1d.day.values]
            vals = da_1d.values
            
            x_dates = [datetime.strptime(d, "%Y-%m-%d") for d in days]
            
            line = plt.plot(x_dates, vals, label=i, linewidth=1)
            
            if connect_gaps:
                valid = [(x, v) for x, v in zip(x_dates, vals) if not np.isnan(v)]
                if valid:
                    x_clean, v_clean = zip(*valid)
                    plt.plot(x_clean, v_clean, color=line[0].get_color(), alpha=0.3, linewidth=1)
        
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
        
        plt.title(f"{', '.join(idents)}: {fVar} ({start_day} to {end_day})", loc='left', fontsize=14)
        plt.xticks(rotation=30, ha='right')
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()