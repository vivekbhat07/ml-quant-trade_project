import yfinance as yf
import pandas as pd

# List of VW supplier tickers
tickers = [
    "CON.DE","BAS.DE","SIE.DE","IFX.DE","SAP.DE","SHA.DE","RHM.DE","SZG.DE",
    "TKA.DE","DUE.DE","HLE.DE","LEO.DE","FR.PA","EO.PA","GEST.MC","SKF-B.ST",
    "NEMAKA.MX","066570.KQ","MGA","JCI","BWA","APTV","TEL"
]

# Create a full date range
full_dates = pd.date_range(start="2005-01-01", end="2020-12-31")

# Dictionary to hold processed data
data_dict = {}

# Download each ticker's data
for ticker in tickers:
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start="2005-01-01", end="2020-12-31")
    
    # Check if 'Adj Close' exists; otherwise use 'Close'
    close_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    
    # Compute daily return based on Close/Adj Close
    df['Return'] = df[close_col].pct_change(fill_method=None)
    
    # Keep only Open, Close, Return
    df = df[['Open', close_col, 'Return']]
    
    # Rename columns to include ticker
    df.columns = [f"{ticker}_Open", f"{ticker}_Close", f"{ticker}_Return"]
    
    # Reindex to full date range and fill missing values with 0
    df = df.reindex(full_dates, fill_value=0)
    
    data_dict[ticker] = df

# Merge all tickers on the index (date)
combined_df = pd.concat(data_dict.values(), axis=1)

# Reset index so Date becomes a column
combined_df.reset_index(inplace=True)
combined_df.rename(columns={'index':'Date'}, inplace=True)

# Save to single CSV
combined_df.to_csv("vw_suppliers_open_close_return_2005_2020_full_fixed.csv", index=False)

print("✅ Done! All tickers' Open, Close, Return saved in one CSV with missing dates filled as 0.")

