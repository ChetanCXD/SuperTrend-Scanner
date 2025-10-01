# SuperTrend S&P 500 Scanner

This project scans all stocks in the S&P 500 index on the daily timeframe (1-day candles) using the SuperTrend indicator applied to Heikin Ashi candles. The scanner identifies and reports stocks that have experienced a trend change on the most recent daily candle.

There are two folders 4h and 1d which are just the same script on different timeframes

## How to Run the Scanner

1.  **Set up the Environment:**

    First, create and activate a Python virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install Dependencies:**

    Install the required Python libraries:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Scanner:**

    Execute the application. It is recommended to run the script about 30 minutes after market close (e.g., 4:30 PM EST) to ensure you have the latest end-of-day data.

    ```bash
    python3 app.py
    ```

## Understanding the Output

After the script runs, it will create a file named `summary.md`. This file contains a detailed report of all stocks that have flipped their SuperTrend trend.

For each flipped ticker, the summary includes:

-   **Date of the Trend Change:** The date of the most recent candle where the flip occurred.
-   **Trend Direction:** The direction of the flip (e.g., `Downtrend -> Uptrend`).
-   **SuperTrend Bands:** The final values of the upper and lower SuperTrend bands.
-   **Recent Raw OHLC Data:** A table showing the last 10 days of raw price data for context.

## Indicator Logic

The scanner faithfully replicates the logic of the following Pine Script, as specified in the original project requirements:

-   **Indicator:** SuperTrend
-   **ATR Period:** 10
-   **ATR Multiplier:** 3.0
-   **Candles:** The ATR and SuperTrend bands are calculated based on Heikin Ashi candles.
-   **Flip Logic:** The trend flip is determined by the raw OHLC close price crossing the SuperTrend bands.

### A Note on Data Discrepancies

This script uses `yfinance` to fetch free stock data from Yahoo! Finance. It is important to note that this data may have slight discrepancies when compared to premium data feeds used by platforms like TradingView or major brokerages (e.g., Robinhood). This can occasionally lead to minor differences in indicator values and signals. The script correctly interprets the data it receives, but the signals are only as accurate as the data source.