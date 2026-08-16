import os
import json
import time
import urllib.request
import numpy as np
from pathlib import Path
from typing import Any

SEC_HEADERS = {'User-Agent': 'GillsQuant research@gillsquant.com'}
CACHE_DIR = Path(__file__).resolve().parent.parent / 'universes'
TICKER_CIK_CACHE_FILE = CACHE_DIR / 'sec_ticker_cik.json'

TAG_MAP = {
    'revenue': [
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'SalesRevenueNet',
        'Revenues',
        'TotalRevenuesAndOtherIncome',
        'RealEstateRevenueNet'
    ],
    'costOfGoodsSold': [
        'CostOfGoodsAndServicesSold',
        'CostOfRevenue',
        'CostOfGoodsSold',
        'CostOfDirectMaterials'
    ],
    'grossProfit': [
        'GrossProfit'
    ],
    'operatingExpenses': [
        'OperatingExpenses',
        'CostsAndExpenses'
    ],
    'researchAndDevelopment': [
        'ResearchAndDevelopmentExpense',
        'ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost'
    ],
    'sellingGeneralAdmin': [
        'SellingGeneralAndAdministrativeExpense',
        'SellingAndMarketingExpense',
        'GeneralAndAdministrativeExpense'
    ],
    'operatingIncome': [
        'OperatingIncomeLoss'
    ],
    'interestExpense': [
        'InterestExpense',
        'InterestAndDebtExpense',
        'InterestExpenseNonoperating'
    ],
    'pretaxIncome': [
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeTaxes',
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest'
    ],
    'incomeTaxExpense': [
        'IncomeTaxExpenseBenefit'
    ],
    'netIncome': [
        'NetIncomeLoss',
        'ProfitLoss'
    ],
    'depreciationAmortization': [
        'DepreciationDepletionAndAmortization',
        'DepreciationAndAmortization',
        'Depreciation'
    ],
    'epsBasic': [
        'EarningsPerShareBasic'
    ],
    'epsDiluted': [
        'EarningsPerShareDiluted'
    ],
    'sharesDiluted': [
        'WeightedAverageNumberOfDilutedSharesOutstanding'
    ],
    'cashAndEquivalents': [
        'CashAndCashEquivalentsAtCarryingValue',
        'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents'
    ],
    'shortTermInvestments': [
        'MarketableSecuritiesCurrent',
        'ShortTermInvestments',
        'AvailableForSaleSecuritiesCurrent'
    ],
    'accountsReceivable': [
        'AccountsReceivableNetCurrent',
        'ReceivablesNetCurrent'
    ],
    'inventory': [
        'InventoryNet',
        'InventoryFinishedGoodsNetOfReserves'
    ],
    'currentAssets': [
        'AssetsCurrent'
    ],
    'ppAndENet': [
        'PropertyPlantAndEquipmentNet'
    ],
    'goodwill': [
        'Goodwill'
    ],
    'intangibleAssets': [
        'IntangibleAssetsNetExcludingGoodwill',
        'FiniteLivedIntangibleAssetsNet'
    ],
    'nonCurrentAssets': [
        'AssetsNoncurrent'
    ],
    'totalAssets': [
        'Assets'
    ],
    'accountsPayable': [
        'AccountsPayableCurrent'
    ],
    'shortTermDebt': [
        'DebtCurrent',
        'CommercialPaper',
        'ShortTermBorrowings'
    ],
    'currentLiabilities': [
        'LiabilitiesCurrent'
    ],
    'longTermDebt': [
        'LongTermDebtNoncurrent',
        'LongTermDebtAndCapitalLeaseObligations'
    ],
    'nonCurrentLiabilities': [
        'LiabilitiesNoncurrent'
    ],
    'totalLiabilities': [
        'Liabilities'
    ],
    'commonStock': [
        'CommonStockValue',
        'CommonStocksIncludingAdditionalPaidInCapital'
    ],
    'retainedEarnings': [
        'RetainedEarningsAccumulatedDeficit'
    ],
    'stockholdersEquity': [
        'StockholdersEquity',
        'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'
    ],
    'operatingCashFlow': [
        'NetCashProvidedByUsedInOperatingActivities'
    ],
    'capitalExpenditures': [
        'PaymentsToAcquirePropertyPlantAndEquipment',
        'PaymentsToAcquireProductiveAssets'
    ],
    'investingCashFlow': [
        'NetCashProvidedByUsedInInvestingActivities'
    ],
    'financingCashFlow': [
        'NetCashProvidedByUsedInFinancingActivities'
    ],
    'dividendsPaid': [
        'PaymentsOfDividends',
        'PaymentsOfDividendsCommonStock'
    ],
    'shareRepurchases': [
        'PaymentsForRepurchaseOfCommonStock',
        'PaymentsForRepurchaseOfEquity'
    ],
    'debtIssuance': [
        'ProceedsFromIssuanceOfLongTermDebt'
    ],
    'debtRepayment': [
        'RepaymentsOfLongTermDebt'
    ],
    'netChangeInCash': [
        'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect'
    ]
}

def get_ticker_cik_map(force_refresh: bool = False) -> dict:
    """
    Downloads and caches the official SEC Ticker -> CIK mapping JSON file.
    Returns dict: {'AAPL': '0000320193', ...}
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    if not force_refresh and TICKER_CIK_CACHE_FILE.exists():
        # Check cache age (refresh if older than 7 days)
        file_age_days = (time.time() - TICKER_CIK_CACHE_FILE.stat().st_mtime) / (3600 * 24)
        if file_age_days < 7:
            try:
                with open(TICKER_CIK_CACHE_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                pass

    try:
        url = 'https://www.sec.gov/files/company_tickers.json'
        req = urllib.request.Request(url, headers=SEC_HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            
        ticker_map = {
            v['ticker'].upper(): str(v['cik_str']).zfill(10)
            for v in data.values()
            if isinstance(v, dict) and v.get('ticker') and v.get('cik_str') is not None
        }
        
        with open(TICKER_CIK_CACHE_FILE, 'w') as f:
            json.dump(ticker_map, f)
            
        return ticker_map
    except Exception as e:
        print(f"Failed to fetch SEC Ticker->CIK map: {e}")
        if TICKER_CIK_CACHE_FILE.exists():
            with open(TICKER_CIK_CACHE_FILE, 'r') as f:
                return json.load(f)
        return {}

def fetch_sec_company_facts(cik: str) -> dict:
    """
    Fetches official SEC XBRL company facts for a given CIK directly in-memory
    without saving any disk cache files.
    """
    if not cik:
        return {}
    cik_padded = str(cik).zfill(10)
    url = f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json'
    req = urllib.request.Request(url, headers=SEC_HEADERS)
    try:
        time.sleep(0.11)  # SEC rate limit: <=10 req/sec
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data
    except Exception:
        return {}

def _safe_float(val, default=np.nan):
    if val is None:
        return default
    try:
        res = float(val)
        return res if not np.isnan(res) else default
    except (ValueError, TypeError):
        return default

def extract_point_in_time_sec_fundamentals(facts: dict, target_day: str) -> dict:
    """
    Parses SEC XBRL company facts for target_day, selecting only records filed on or before target_day.
    Returns a dictionary of fundamental variables keyed by name.
    """
    target_clean = str(target_day)[:10]
    if not facts or 'facts' not in facts or 'us-gaap' not in facts['facts']:
        return {}

    gaap = facts['facts']['us-gaap']
    extracted = {}
    
    latest_filed = None
    latest_end = None
    
    # 1. Extract base fields using (end, filed) priority
    for key, tag_list in TAG_MAP.items():
        best_item = None
        for tag in tag_list:
            if tag in gaap:
                units = gaap[tag].get('units', {})
                for unit_name in ['USD', 'shares', 'USD/shares']:
                    if unit_name in units:
                        for item in units[unit_name]:
                            filed = str(item.get('filed') or '')
                            form = str(item.get('form') or '')
                            end = str(item.get('end') or '')
                            if form in ('10-K', '10-Q', '10-K/A', '10-Q/A') and filed and filed <= target_clean:
                                if best_item is None or (end, filed) > (str(best_item.get('end') or ''), str(best_item.get('filed') or '')):
                                    best_item = item
        if best_item is not None:
            extracted[key] = _safe_float(best_item.get('val'), default=np.nan)
            f_date = str(best_item.get('filed') or '').replace('-', '')
            e_date = str(best_item.get('end') or '').replace('-', '')
            if f_date.isdigit() and (latest_filed is None or f_date > latest_filed):
                latest_filed = f_date
            if e_date.isdigit() and (latest_end is None or e_date > latest_end):
                latest_end = e_date

    if not extracted:
        return {}

    # Metadata fields
    extracted['filingDate'] = _safe_float(latest_filed) if latest_filed else np.nan
    extracted['periodEndDate'] = _safe_float(latest_end) if latest_end else np.nan

    # Derived accounting fields
    rev = extracted.get('revenue', np.nan)
    cogs = extracted.get('costOfGoodsSold', np.nan)
    op_inc = extracted.get('operatingIncome', np.nan)
    net_inc = extracted.get('netIncome', np.nan)
    da = _safe_float(extracted.get('depreciationAmortization'), default=0.0)
    assets = extracted.get('totalAssets', np.nan)
    liab = extracted.get('totalLiabilities', np.nan)
    equity = extracted.get('stockholdersEquity', np.nan)
    curr_assets = extracted.get('currentAssets', np.nan)
    curr_liab = extracted.get('currentLiabilities', np.nan)
    st_debt = _safe_float(extracted.get('shortTermDebt'), default=0.0)
    lt_debt = _safe_float(extracted.get('longTermDebt'), default=0.0)
    cash = _safe_float(extracted.get('cashAndEquivalents'), default=0.0)
    st_inv = _safe_float(extracted.get('shortTermInvestments'), default=0.0)
    rec = _safe_float(extracted.get('accountsReceivable'), default=0.0)
    op_cf = extracted.get('operatingCashFlow', np.nan)
    capex = _safe_float(extracted.get('capitalExpenditures'), default=0.0)

    # Gross Profit derivation
    if np.isnan(extracted.get('grossProfit', np.nan)) and not np.isnan(rev) and not np.isnan(cogs):
        extracted['grossProfit'] = rev - cogs

    # EBITDA derivation
    if not np.isnan(op_inc):
        extracted['ebitda'] = op_inc + da

    # Net Debt
    tot_debt = st_debt + lt_debt
    extracted['netDebt'] = tot_debt - (cash + st_inv)

    # Free Cash Flow
    if not np.isnan(op_cf):
        extracted['freeCashFlow'] = op_cf - capex

    # Helper for ratio division
    def _safe_ratio(num, den):
        if num is None or den is None or np.isnan(num) or np.isnan(den) or den == 0:
            return np.nan
        return num / den

    # Derived Ratios
    extracted['grossMargin'] = _safe_ratio(extracted.get('grossProfit'), rev)
    extracted['operatingMargin'] = _safe_ratio(op_inc, rev)
    extracted['profitMargin'] = _safe_ratio(net_inc, rev)
    extracted['roe'] = _safe_ratio(net_inc, equity)
    extracted['roa'] = _safe_ratio(net_inc, assets)
    extracted['currentRatio'] = _safe_ratio(curr_assets, curr_liab)
    extracted['quickRatio'] = _safe_ratio((cash + st_inv + rec), curr_liab)
    extracted['debtToEquity'] = _safe_ratio(tot_debt, equity)

    return extracted


CURRENT_EDGAR_PARQUET_FILE = Path(__file__).resolve().parent.parent / 'data' / 'current' / 'current_edgar_data.parquet'

def update_ticker_cik_map(max_retries: int = 5) -> dict:
    """
    Fetches SEC ticker -> CIK map with up to max_retries retries and exponential backoff.
    Executed nightly at 03:15 AM.
    """
    for attempt in range(1, max_retries + 1):
        try:
            url = 'https://www.sec.gov/files/company_tickers.json'
            req = urllib.request.Request(url, headers=SEC_HEADERS)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            ticker_map = {
                v['ticker'].upper(): str(v['cik_str']).zfill(10)
                for v in data.values()
                if isinstance(v, dict) and v.get('ticker') and v.get('cik_str') is not None
            }
            with open(TICKER_CIK_CACHE_FILE, 'w') as f:
                json.dump(ticker_map, f)
            print(f"\tUpdated SEC Ticker->CIK map ({len(ticker_map)} tickers)", flush=True)
            return ticker_map
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                print(f"\tFailed fetching SEC Ticker->CIK map: {e}", flush=True)
    return get_ticker_cik_map(force_refresh=False)

TODAYS_FILING_SYMBOLS_FILE = CACHE_DIR / 'todays_filing_symbols.json'

def detect_todays_filing_symbols(universe_symbols: list = None, max_retries: int = 5) -> list:
    """
    Polls SEC Atom RSS feed & Schwab earnings dates to find symbols requiring updates today.
    Saves findings to universes/todays_filing_symbols.json for the 03:25 AM task.
    Executed nightly at 03:20 AM.
    """
    import polars as pl
    if universe_symbols is None:
        try:
            u_df = pl.read_csv(CACHE_DIR / 'u00.csv')
            universe_symbols = u_df['name'].to_list()
        except Exception:
            universe_symbols = []

    ticker_cik_map = get_ticker_cik_map()
    ciks_today = set()
    
    for attempt in range(1, max_retries + 1):
        try:
            for form_type in ['10-Q', '10-K']:
                url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type={form_type}&count=100&output=atom"
                req = urllib.request.Request(url, headers=SEC_HEADERS)
                with urllib.request.urlopen(req, timeout=8) as resp:
                    xml_data = resp.read()
                import xml.etree.ElementTree as ET
                root = ET.fromstring(xml_data)
                for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                    title = entry.find('{http://www.w3.org/2005/Atom}title')
                    if title is not None and title.text:
                        import re
                        match = re.search(r'\((\d{10})\)', title.text)
                        if match:
                            ciks_today.add(match.group(1))
            break
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                print(f"\tFailed polling SEC RSS feed: {e}", flush=True)
                
    symbols_to_update = []
    if universe_symbols:
        for sym in universe_symbols:
            cik = ticker_cik_map.get(sym.upper())
            if cik and cik in ciks_today:
                symbols_to_update.append(sym)
                
    try:
        with open(TODAYS_FILING_SYMBOLS_FILE, 'w') as f:
            json.dump(symbols_to_update, f)
    except Exception as e:
        print(f"\tWarning: Failed writing {TODAYS_FILING_SYMBOLS_FILE.name}: {e}")

    return symbols_to_update

def rebuild_current_edgar_data_file(universe_symbols: list = None, max_retries: int = 5) -> Any:
    """
    Alternate recovery method to replace a lost, missing, or corrupted data/current/current_edgar_data.parquet file.
    Streams SEC facts directly in-memory for universe u00 symbols
    and writes a fresh data/current/current_edgar_data.parquet file (zero raw JSON files saved to disk).
    """
    import polars as pl

    if universe_symbols is None:
        try:
            u_df = pl.read_csv(CACHE_DIR / 'u00.csv')
            universe_symbols = u_df['name'].to_list()
        except Exception:
            universe_symbols = []

    print(f"[EDGAR Recovery] Attempting to rebuild missing {CURRENT_EDGAR_PARQUET_FILE}...", flush=True)

    # Rebuild directly in-memory without saving any raw JSON cache files
    ticker_cik_map = get_ticker_cik_map()
    target_day = time.strftime("%Y-%m-%d")
    
    rows = []
    print(f"[EDGAR Recovery] Streaming in-memory SEC facts for {len(universe_symbols)} universe symbols...", flush=True)
    for idx, sym in enumerate(universe_symbols):
        cik = ticker_cik_map.get(sym.upper())
        if cik:
            for attempt in range(1, max_retries + 1):
                try:
                    facts = fetch_sec_company_facts(cik)
                    if facts:
                        sec_dict = extract_point_in_time_sec_fundamentals(facts, target_day)
                        if sec_dict:
                            sec_dict['ident'] = sym
                            rows.append(sec_dict)
                        del facts
                    break
                except Exception as e:
                    if attempt == max_retries:
                        pass
                    else:
                        time.sleep(1)

    df = pl.DataFrame(rows) if rows else pl.DataFrame()
    if not df.is_empty():
        CURRENT_EDGAR_PARQUET_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(CURRENT_EDGAR_PARQUET_FILE)
        print(f"✓ [EDGAR Recovery] Successfully rebuilt {CURRENT_EDGAR_PARQUET_FILE.name} ({len(df)} symbols, size: {CURRENT_EDGAR_PARQUET_FILE.stat().st_size / 1024:.1f} KB)!", flush=True)
    return df

def update_current_edgar_data_file(symbols_to_update: list = None, universe_symbols: list = None, max_retries: int = 5) -> Any:
    """
    Updates data/current_edgar_data.parquet for symbols_to_update,
    trims obsolete tickers not in universe_symbols, and adds entries for new symbols.
    Executed nightly at 03:25 AM.
    """
    import polars as pl
    if universe_symbols is None:
        try:
            u_df = pl.read_csv(CACHE_DIR / 'u00.csv')
            universe_symbols = u_df['name'].to_list()
        except Exception:
            universe_symbols = []

    if symbols_to_update is None and TODAYS_FILING_SYMBOLS_FILE.exists():
        try:
            with open(TODAYS_FILING_SYMBOLS_FILE, 'r') as f:
                symbols_to_update = json.load(f)
        except Exception:
            symbols_to_update = []

    if not CURRENT_EDGAR_PARQUET_FILE.exists():
        return rebuild_current_edgar_data_file(universe_symbols, max_retries=max_retries)

    current_df = pl.read_parquet(CURRENT_EDGAR_PARQUET_FILE)
    ticker_cik_map = get_ticker_cik_map()
    target_day = time.strftime("%Y-%m-%d")
    
    # 1. Fetch & extract metrics for symbols_to_update
    updates_dict = {}
    if symbols_to_update:
        for sym in symbols_to_update:
            cik = ticker_cik_map.get(sym.upper())
            if cik:
                for attempt in range(1, max_retries + 1):
                    try:
                        facts = fetch_sec_company_facts(cik)
                        sec_dict = extract_point_in_time_sec_fundamentals(facts, target_day)
                        if sec_dict:
                            sec_dict['ident'] = sym
                            updates_dict[sym] = sec_dict
                        break
                    except Exception as e:
                        if attempt == max_retries:
                            print(f"[03:25 EDGAR Update] Failed fetching facts for {sym}: {e}", flush=True)
                        else:
                            time.sleep(2 ** attempt)

    # 2. Trim symbols no longer in universe_symbols
    if universe_symbols and 'ident' in current_df.columns:
        current_df = current_df.filter(pl.col('ident').is_in(universe_symbols))
        
    # 3. Upsert updates and add new symbols
    existing_idents = set(current_df['ident'].to_list()) if 'ident' in current_df.columns else set()
    new_rows = []
    
    for sym, row in updates_dict.items():
        if sym in existing_idents:
            current_df = current_df.filter(pl.col('ident') != sym)
        new_rows.append(row)

    if new_rows:
        new_df = pl.DataFrame(new_rows)
        current_df = pl.concat([current_df, new_df], how='diagonal_relaxed')
        
    CURRENT_EDGAR_PARQUET_FILE.parent.mkdir(parents=True, exist_ok=True)
    current_df.write_parquet(CURRENT_EDGAR_PARQUET_FILE)
    print(f"\tUpdated {CURRENT_EDGAR_PARQUET_FILE.name} (Total Symbols: {len(current_df)})", flush=True)
    return current_df

def read_current_edgar_data() -> Any:
    """Reads data/current_edgar_data.parquet locally in <5 milliseconds. Auto-rebuilds if missing."""
    import polars as pl
    if not CURRENT_EDGAR_PARQUET_FILE.exists():
        return rebuild_current_edgar_data_file()
    try:
        return pl.read_parquet(CURRENT_EDGAR_PARQUET_FILE)
    except Exception as e:
        print(f"Warning: Failed reading {CURRENT_EDGAR_PARQUET_FILE.name}: {e}")
        return rebuild_current_edgar_data_file()


