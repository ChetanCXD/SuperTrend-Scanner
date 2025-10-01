#!/home/chetan/AlgoBot/venv/bin/python

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import time
from multiprocessing import Pool
from datetime import datetime

# Suppress FutureWarning from yfinance
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_sp500_tickers():
    """
    This function is responsible for returning the list of S&P 500 tickers.
    """
    return [
        "NVDA", "MSFT", "AAPL", "GOOG", "GOOGL", "AMZN", "META", "AVGO", "TSLA", 
        "JPM", "WMT", "ORCL", "V", "LLY", "MA", "NFLX", "XOM", "JNJ", "PLTR", "COST", 
        "HD", "ABBV", "BAC", "PG", "CVX", "UNH", "GE", "KO", "WFC", "TMUS", "CSCO", 
        "AMD", "IBM", "MS", "GS", "PM", "APP", "AXP", "CRM", "ABT", "LIN", "CAT", "RTX", 
        "BX", "MCD", "UBER", "DIS", "T", "MRK", "NOW", "INTU", "PEP", "C", "MU", "VZ", 
        "BLK", "ANET", "QCOM", "BKNG", "SCHW", "TMO", "TXN", "LRCX", "BA", "GEV", "AMAT", 
        "INTC", "TJX", "ISRG", "NEE", "ACN", "ADBE", "SPGI", "APH", "AMGN", "BSX", "PGR", 
        "COF", "ETN", "LOW", "SYK", "KLAC", "UNP", "GILD", "PANW", "PFE", "DHR", "HON", 
        "DE", "CRWD", "HOOD", "MDT", "ADI", "KKR", "COP", "ADP", "WELL", "DASH", "CMCSA", 
        "LMT", "CB", "MO", "PLD", "CEG", "SO", "NKE", "VRTX", "HCA", "MMC", "CME", "SBUX", 
        "ICE", "CVS", "DUK", "PH", "CDNS", "MCK", "NEM", "TT", "ORLY", "AMT", "BMY", 
        "DELL", "SNPS", "GD", "RCL", "WM", "MCO", "COIN", "SHW", "NOC", "CTAS", "MMM", 
        "MDLZ", "PNC", "APO", "AJG", "WMB", "ECL", "BK", "HWM", "CI", "EQIX", "AON", 
        "USB", "ITW", "MSI", "ABNB", "EMR", "TDG", "MAR", "ELV", "UPS", "RSG", "AZO", 
        "FI", "JCI", "SPG", "GLW", "ADSK", "NSC", "VST", "CSX", "PYPL", "WDAY", "MNST", 
        "CL", "FTNT", "TEL", "ZTS", "KMI", "TRV", "EOG", "HLT", "PWR", "URI", "APD", 
        "MPC", "COR", "AFL", "TFC", "DLR", "AEP", "SRE", "GM", "REGN", "CMI", "NXPI", 
        "AXON", "FAST", "FDX", "ALL", "PSX", "LHX", "O", "MET", "ROP", "CMG", "VLO", 
        "FCX", "BDX", "SLB", "PCAR", "D", "NDAQ", "PSA", "DDOG", "DHI", "EA", "IDXX", 
        "CARR", "BKR", "ROST", "STX", "F", "TTWO", "OXY", "XEL", "GRMN", "AMP", "WBD", 
        "CBRE", "PAYX", "XYZ", "OKE", "CTVA", "GWW", "EW", "EXC", "KR", "MSCI", "CPRT", 
        "AME", "AIG", "YUM", "MPWR", "CHTR", "CCI", "ETR", "EBAY", "FANG", "PEG", "TKO", 
        "KMB", "WDC", "TGT", "VMC", "RMD", "SYY", "ROK", "CCL", "LYV", "LVS", "MLM", 
        "DAL", "HSY", "HIG", "WEC", "CAH", "FICO", "PRU", "TRGP", "ED", "OTIS", "CSGP", 
        "RJF", "A", "XYL", "KDP", "VRSK", "VICI", "MCHP", "FIS", "EQT", "WAB", "ACGL", 
        "WTW", "GEHC", "STT", "PCG", "LEN", "IR", "CTSH", "DD", "EL", "NRG", "UAL", 
        "HPE", "VTR", "EFX", "KVUE", "EXR", "NUE", "MTB", "IQV", "HUM", "BRO", "IBKR", 
        "KHC", "FITB", "TSCO", "KEYS", "ODFL", "IRM", "DTE", "ADM", "WRB", "EME", "K", 
        "ROL", "FOXA", "AEE", "BR", "AWK", "SMCI", "AVB", "PPL", "EXPE", "SYF", "ATO", 
        "TDY", "VRSN", "GIS", "PHM", "FE", "ES", "VLTO", "DXCM", "FOX", "CBOE", "CNP", 
        "NTRS", "HBAN", "EXE", "EQR", "HPQ", "ULTA", "MTD", "CINF", "PTC", "IP", "TTD", 
        "STE", "STZ", "FSLR", "LDOS", "NTAP", "RF", "LH", "PPG", "WSM", "NVR", "CFG", 
        "TPR", "JBL", "TYL", "DOV", "TROW", "HUBB", "DG", "DVN", "DRI", "SW", "PODD", 
        "CMS", "PSKY", "TER", "HAL", "CDW", "EIX", "LULU", "TPL", "DGX", "SBAC", "CHD", 
        "GPN", "CPAY", "KEY", "STLD", "ON", "NI", "BIIB", "IT", "GDDY", "TRMB", "ZBH", "GP"
    ]

def get_data(ticker, retries=5, delay=10, timeout=30):
    """
    This function is responsible for fetching the historical data for a given ticker with retry logic.
    """
    for i in range(retries):
        try:
            return yf.download(ticker, period="6mo", interval="1d", progress=False, timeout=timeout)
        except Exception as e:
            if i < retries - 1:
                time.sleep(delay * (i + 1)) # Exponential backoff
            else:
                # Suppress final error message for cleaner output
                pass
    return None

def heikin_ashi(df):
    """
    This function is responsible for converting the OHLC data to Heikin Ashi candles.
    """
    ha_df = df.copy()
    ha_df['Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

    for i in range(len(df)):
        if i == 0:
            ha_df.at[ha_df.index[i], 'Open'] = (df['Open'].iloc[i] + df['Close'].iloc[i]) / 2
        else:
            ha_df.at[ha_df.index[i], 'Open'] = (ha_df.at[ha_df.index[i-1], 'Open'] + ha_df.at[ha_df.index[i-1], 'Close']) / 2

    ha_df['High'] = ha_df[['Open', 'Close']].join(df['High']).max(axis=1)
    ha_df['Low'] = ha_df[['Open', 'Close']].join(df['Low']).min(axis=1)

    return ha_df

def atr(df, period):
    """
    This function is responsible for calculating the Average True Range (ATR) using RMA, as specified in the Pine Script.
    """
    tr = pd.DataFrame(index=df.index)
    tr['tr0'] = abs(df['High'] - df['Low'])
    tr['tr1'] = abs(df['High'] - df['Close'].shift(1))
    tr['tr2'] = abs(df['Low'] - df['Close'].shift(1))
    tr['tr'] = tr[['tr0', 'tr1', 'tr2']].max(axis=1)
    # The `atr` function in Pine Script uses RMA (Relative Moving Average)
    # which is equivalent to an exponentially weighted moving average with alpha = 1/length
    atr = tr['tr'].ewm(alpha=1/period, adjust=False).mean()
    return atr

def supertrend(df, ha_df, period, multiplier):
    """
    Calculates the SuperTrend indicator based on the exact Pine Script logic.

    Args:
        df (pd.DataFrame): The raw OHLC data.
        ha_df (pd.DataFrame): The Heikin Ashi converted OHLC data.
        period (int): The ATR period (default 10).
        multiplier (float): The ATR multiplier (default 3.0).

    Returns:
        pd.DataFrame: A DataFrame with 'trend', 'up', and 'down' columns.
    """
    # As per the Pine Script, ATR is calculated on Heikin Ashi candles.
    atr_val = atr(ha_df, period)
    
    # The 'src' in the Pine Script is hl2, which is (high + low) / 2. 
    # We calculate this on the Heikin Ashi data.
    src = (ha_df['High'] + ha_df['Low']) / 2
    
    # Initial upper and lower bands are calculated using the Heikin Ashi-based src and ATR.
    up = src - (multiplier * atr_val)
    dn = src + (multiplier * atr_val)
    
    # Initialize the trend series with NaN
    trend = pd.Series(np.nan, index=df.index)
    trend.iloc[0] = 1 # Default to an uptrend for the first candle
    
    # Process bar by bar to accurately replicate the recursive Pine Script logic
    for i in range(1, len(df)):
        # Pine: up := close[1] > up1 ? max(up,up1) : up
        # If the previous RAW close was above the previous upper band, the new upper band is the max of the current and previous upper bands.
        if df['Close'].iloc[i-1] > up.iloc[i-1]:
            up.iloc[i] = max(up.iloc[i], up.iloc[i-1])
        
        # Pine: dn := close[1] < dn1 ? min(dn, dn1) : dn
        # If the previous RAW close was below the previous lower band, the new lower band is the min of the current and previous lower bands.
        if df['Close'].iloc[i-1] < dn.iloc[i-1]:
            dn.iloc[i] = min(dn.iloc[i], dn.iloc[i-1])

        # Pine: trend := trend == -1 and close > dn1 ? 1 : trend == 1 and close < up1 ? -1 : trend
        # If the trend was a downtrend (-1) and the current RAW close crosses ABOVE the previous lower band, flip to an uptrend (1).
        if trend.iloc[i-1] == -1 and df['Close'].iloc[i] > dn.iloc[i-1]:
            trend.iloc[i] = 1
        # If the trend was an uptrend (1) and the current RAW close crosses BELOW the previous upper band, flip to a downtrend (-1).
        elif trend.iloc[i-1] == 1 and df['Close'].iloc[i] < up.iloc[i-1]:
            trend.iloc[i] = -1
        # Otherwise, the trend continues from the previous bar.
        else:
            trend.iloc[i] = trend.iloc[i-1]
            
    st = pd.DataFrame(index=df.index)
    st['trend'] = trend
    st['up'] = up
    st['down'] = dn
    
    return st

def process_ticker(ticker):
    """
    This function is responsible for processing a single ticker and returning the flip status.
    """
    data = get_data(ticker)
    if data is None or data.empty or len(data) < 100:
        return None

    # Drop the multi-level column index
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    # 1. Convert to Heikin Ashi for ATR and band calculations
    ha_data = heikin_ashi(data)

    # 2. Calculate SuperTrend using the accurate hybrid model
    st = supertrend(data, ha_data, 10, 3.0)

    if st is None or st.empty:
        return None

    # Check for flip on the most recent candle
    if len(st['trend']) > 1 and st['trend'].iloc[-1] != st['trend'].iloc[-2]:
        prev_trend_val = st['trend'].iloc[-2]
        curr_trend_val = st['trend'].iloc[-1]

        prev_trend = "Uptrend" if prev_trend_val == 1 else "Downtrend"
        curr_trend = "Uptrend" if curr_trend_val == 1 else "Downtrend"

        return {
            "ticker": ticker,
            "date": data.index[-1].date(),
            "previous_trend": prev_trend,
            "new_trend": curr_trend,
            "supertrend_up": st['up'].iloc[-1],
            "supertrend_down": st['down'].iloc[-1],
            "data": data.tail(10) # Return raw data for context
        }
    else:
        return None

if __name__ == "__main__":
    tickers = get_sp500_tickers()
    with Pool(processes=4) as pool:
        results = pool.map(process_ticker, tickers)
    
    flipped_tickers = [result for result in results if result]
    
    md_content = "# SuperTrend Scanner Summary\n\n"
    md_content += f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    md_content += "This report identifies stocks from the S&P 500 that have experienced a SuperTrend trend change on the most recent daily candle. The calculation is based on the Pine Script logic from the project's README, using a 10-period ATR with a multiplier of 3.\n\n---\n"

    if flipped_tickers:
        for result in flipped_tickers:
            md_content += f"### {result['ticker']}: Trend Change Detected\n"
            md_content += f"- **Date of Change:** {result['date']}\n"
            md_content += f"- **Trend Change:** `{result['previous_trend']} -> {result['new_trend']}`\n"
            md_content += f"- **SuperTrend Upper Band:** `{result['supertrend_up']:.2f}`\n"
            md_content += f"- **SuperTrend Lower Band:** `{result['supertrend_down']:.2f}`\n\n"
            md_content += "**Recent Raw OHLC Data:**\n"
            md_content += result['data'].to_markdown() + "\n\n---\n\n"
    else:
        md_content += "**No tickers experienced a SuperTrend trend change on the most recent daily candle.**\n"

    with open("summary1d.md", "w") as f:
        f.write(md_content)

    print("Summary saved to summary1d.md")
