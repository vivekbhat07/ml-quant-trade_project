import pandas as pd

# Load both CSVs
df1 = pd.read_csv("volkswagen_stock_2005_2020.csv")   # contains 'Price' and 'return_pct'
df2 = pd.read_csv("mlQuantTrade.csv")  # contains 'Date' and stock data

# Ensure date columns match format
df1['Price'] = pd.to_datetime(df1['Price'])
df2['Date'] = pd.to_datetime(df2['Date'])

# Merge on date
merged_df = pd.merge(df2, df1[['Price', 'return_pct']], 
                     left_on='Date', right_on='Price', how='left')

# Drop duplicate 'Price' column (since it's same as Date)
merged_df.drop(columns=['Price'], inplace=True)

# Save result
merged_df.to_csv("merged.csv", index=False)

