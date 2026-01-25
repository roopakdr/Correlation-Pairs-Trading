import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import logging
import requests
from io import StringIO
from tabulate import tabulate
import warnings

# Suppress warnings and yfinance noise
warnings.filterwarnings('ignore', category=RuntimeWarning)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

def get_sp500_data():
    """Get S&P 500 stocks with sectors from Wikipedia."""
    print("--- Fetching S&P 500 data from Wikipedia ---")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    sp500_df = pd.read_html(StringIO(response.text))[0]
    
    # Create sector map for S&P 500
    sp500_df['Symbol'] = sp500_df['Symbol'].str.replace('.', '-')
    sector_map = dict(zip(sp500_df['Symbol'], sp500_df['GICS Sector']))
    
    print(f"Found {len(sector_map)} S&P 500 stocks")
    return list(sector_map.keys()), sector_map

def get_nasdaq_data():
    """Get NASDAQ screener data."""
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
        'accept': 'application/json, text/plain, */*',
    }
    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=0&download=true"
    
    print("--- Fetching NASDAQ screener data ---")
    response = requests.get(url, headers=headers, timeout=15)
    data = response.json()
    rows = data['data']['rows']
    df = pd.DataFrame(rows)
    
    df['symbol'] = df['symbol'].str.replace('/', '-').str.replace('^', '-')
    sector_map = dict(zip(df['symbol'], df['sector']))
    
    print(f"Found {len(sector_map)} NASDAQ screener stocks")
    return list(df['symbol'].unique()), sector_map

def standardize_sector(sector):
    """Map NASDAQ sector names to standard GICS sectors."""
    if not sector or sector.strip() == "":
        return None
    
    # Mapping of NASDAQ sectors to GICS sectors
    sector_map = {
        # Technology variants
        'technology': 'Information Technology',
        'computer and technology': 'Information Technology',
        'telecommunications': 'Communication Services',
        'media': 'Communication Services',
        
        # Finance variants
        'finance': 'Financials',
        'financial services': 'Financials',
        'banks': 'Financials',
        'insurance': 'Financials',
        
        # Consumer variants
        'consumer services': 'Consumer Discretionary',
        'consumer goods': 'Consumer Staples',
        'retail': 'Consumer Discretionary',
        
        # Health variants
        'health care': 'Health Care',
        'healthcare': 'Health Care',
        
        # Materials variants
        'basic materials': 'Materials',
        'basic industries': 'Materials',
        
        # Industrial variants
        'capital goods': 'Industrials',
        'transportation': 'Industrials',
        
        # Energy variants
        'energy': 'Energy',
        'oil & gas': 'Energy',
        
        # Utilities variants
        'utilities': 'Utilities',
        'public utilities': 'Utilities',
        
        # Real Estate variants
        'real estate': 'Real Estate',
    }
    
    sector_lower = sector.lower().strip()
    standardized = sector_map.get(sector_lower, None)
    
    # Only return valid GICS sectors (filter out Miscellaneous, etc.)
    valid_gics_sectors = {
        'Communication Services',
        'Consumer Discretionary', 
        'Consumer Staples',
        'Energy',
        'Financials',
        'Health Care',
        'Industrials',
        'Information Technology',
        'Materials',
        'Real Estate',
        'Utilities'
    }
    
    if standardized in valid_gics_sectors:
        return standardized
    
    # If not mapped and not a valid GICS sector, return None (exclude it)
    return None

def get_full_market_data():
    """Combine S&P 500 and NASDAQ data with standardized sectors."""
    # Get both datasets
    sp500_tickers, sp500_sectors = get_sp500_data()
    nasdaq_tickers, nasdaq_sectors = get_nasdaq_data()
    
    # Standardize NASDAQ sectors to GICS
    nasdaq_sectors_standardized = {
        ticker: standardize_sector(sector) 
        for ticker, sector in nasdaq_sectors.items()
    }
    
    # Combine sector maps (S&P 500 takes priority for overlaps)
    combined_sectors = nasdaq_sectors_standardized.copy()
    combined_sectors.update(sp500_sectors)  # S&P 500 sectors override NASDAQ
    
    # Remove stocks with no valid sector
    combined_sectors = {k: v for k, v in combined_sectors.items() if v}
    
    # Combine ticker lists (only include stocks with valid sectors)
    all_tickers = list(combined_sectors.keys())
    
    print()
    print(f"--- Combined Dataset ---")
    print(f"Total unique stocks: {len(all_tickers)}")
    print(f"S&P 500 stocks: {len(sp500_tickers)}")
    print(f"NASDAQ stocks: {len(nasdaq_tickers)}")
    print(f"Overlap: {len(set(sp500_tickers) & set(nasdaq_tickers))}")
    print(f"Unique sectors: {len(set(combined_sectors.values()))}")

    return all_tickers, combined_sectors

all_tickers, sector_map = get_full_market_data()

unique_sectors = sorted(list(set([s for s in sector_map.values() if s and s.strip() != ""])))

print("\nAvailable Sectors found in Nasdaq/S&P Universe:")
for i, s in enumerate(unique_sectors):
    print(f"{i}. {s}") 

choice = input("\nEnter a Sector Name or Index Number: ").strip()

while True:
    if choice.isdigit() and int(choice) < len(unique_sectors):
        target_sector = unique_sectors[int(choice)]
    else:
        target_sector = next((s for s in unique_sectors if s.lower() == choice.lower()), None)
    
    if not target_sector:
        choice = input("Sector not recognized. Please try again: ").strip()
    else:
        break

selected_stocks = [t for t in all_tickers if sector_map.get(t) == target_sector]
print(f"--- Found {len(selected_stocks)} stocks in {target_sector} ---")

stocks_data = yf.download(selected_stocks, period='1y', interval='1d', progress=True)

if 'Adj Close' in stocks_data.columns:
    close_prices = stocks_data['Adj Close']
elif 'Close' in stocks_data.columns:
    close_prices = stocks_data['Close']
else:
    # Handle specific yfinance MultiIndex structures
    try:
        close_prices = stocks_data.xs('Adj Close', axis=1, level=0)
    except:
        close_prices = stocks_data.xs('Close', axis=1, level=0)

# Filter for liquidity
close_prices = close_prices.dropna(axis=1, thresh=240)
print(f'--- Analyzing {len(close_prices.columns)} liquid stocks using Adjusted Prices ---')

pairs = list(combinations(close_prices.columns, 2))
table = []

for stock1, stock2 in pairs:
    df1, df2 = close_prices[stock1].dropna(), close_prices[stock2].dropna()
    common_index = df1.index.intersection(df2.index)
    
    if len(common_index) < 60: continue

    df1_aligned, df2_aligned = df1[common_index], df2[common_index]
    df1_aligned = np.log(df1_aligned)
    df2_aligned = np.log(df2_aligned)
    if df1_aligned.std() == 0 or df2_aligned.std() == 0: continue
    
    daily_change_1 = df1_aligned.diff().dropna()
    daily_change_2 = df2_aligned.diff().dropna()
    correlation = daily_change_1.corr(daily_change_2)

    if correlation > 0.7:
        score, pvalue, _ = coint(df1_aligned, df2_aligned)
        if pvalue < 0.05:
            x = sm.add_constant(df2_aligned)
            model = sm.OLS(df1_aligned, x).fit()
            if len(model.params) < 2: continue
            
            hedge_ratio = model.params.iloc[1]
            spread = df1_aligned - (hedge_ratio * df2_aligned)
            result = adfuller(spread.dropna())
            adf_pvalue = result[1]

            if adf_pvalue >= 0.05: continue  # Not stationary

            spread_lag = spread.shift(1)
            spread_diff = spread - spread_lag
            spread_lag = spread_lag.dropna()
            spread_diff = spread_diff.dropna()
            valid_idx = spread_diff.dropna().index
            res = sm.OLS(spread_diff.loc[valid_idx], sm.add_constant(spread_lag.loc[valid_idx])).fit()
            
            lambda_val = res.params.iloc[1]
            half_life = -np.log(2) / lambda_val if lambda_val < 0 else 999
            
            if 2 < half_life < 30:
                rolling_mean = spread.rolling(window=30, min_periods=20).mean()
                rolling_std = spread.rolling(window=30, min_periods=20).std()
                z_score = (spread - rolling_mean) / rolling_std.replace(0, float('nan'))
            
                z_vals = z_score.dropna()
                if z_vals.empty: continue
                z_score_last = z_vals.iloc[-1]

                if abs(z_score_last) > 2.0:
                    table.append([f'{stock1} & {stock2}', f'{z_score_last:.2f}'])

if table:
    # Sort by absolute Z-Score descending (most extreme first)
    table.sort(key=lambda x: abs(float(x[1])), reverse=True)
    print("\n" + tabulate(table, headers=['Pair', 'Z-Score'], tablefmt='github'))
else:
    print()
    print(f"No suitable pairs found in the {target_sector} sector.")