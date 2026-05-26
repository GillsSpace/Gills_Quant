#Imports
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_zarr_day(zarr_store, day, qVar='quote.mark', ident='AAPL', connect_gaps=False):
    """Extracts 5m data and plots it with a cleaned-up x-axis for one or more idents, handling gaps."""
    idents = [ident] if isinstance(ident, str) else ident
    
    plt.figure(figsize=(12, 5))
    
    for i in idents:
        # Extract data (contains NaNs for missing intervals)
        data_raw = zarr_store['5m'].sel(day=day, qVar=qVar, ident=i).to_pandas()
        
        # Combine day and time into valid datetimes for correct gap rendering
        x_dates = pd.to_datetime([f"{day} {t}" for t in data_raw.index])
        
        # Plot main line with gaps
        line = plt.plot(x_dates, data_raw.values, label=i, linewidth=1.5)
        
        # Conditionally bridge gaps with a transparent line
        if connect_gaps:
            data_clean = data_raw.dropna()
            x_clean = pd.to_datetime([f"{day} {t}" for t in data_clean.index])
            plt.plot(x_clean, data_clean.values, color=line[0].get_color(), alpha=0.3, linewidth=1.5)
    
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
            data_raw = zarr_store['5m'].sel(
                day=slice(start_day, end_day), 
                qVar=qVar, 
                ident=i
            ).stack(timeline=('day', 'time')).to_pandas()
            
            x_dates = pd.to_datetime([f"{d} {t}" for d, t in data_raw.index])
            
            # Plot main line with NaNs (creates gaps)
            line = plt.plot(x_dates, data_raw.values, label=i, linewidth=1)
            
            # Conditionally plot continuous line to bridge gaps
            if connect_gaps:
                data_clean = data_raw.dropna()
                x_clean = pd.to_datetime([f"{d} {t}" for d, t in data_clean.index])
                plt.plot(x_clean, data_clean.values, color=line[0].get_color(), alpha=0.3, linewidth=1)
        
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
            data_raw = zarr_store['1d'].sel(
                day=slice(start_day, end_day), 
                fVar=fVar, 
                ident=i
            ).to_pandas()
            
            x_dates = pd.to_datetime(data_raw.index)
            
            line = plt.plot(x_dates, data_raw.values, label=i, linewidth=1)
            
            if connect_gaps:
                data_clean = data_raw.dropna()
                x_clean = pd.to_datetime(data_clean.index)
                plt.plot(x_clean, data_clean.values, color=line[0].get_color(), alpha=0.3, linewidth=1)
        
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
        
        plt.title(f"{', '.join(idents)}: {fVar} ({start_day} to {end_day})", loc='left', fontsize=14)
        plt.xticks(rotation=30, ha='right')
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()