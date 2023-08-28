# This is the start of MarketEnvelopePro 
import numpy as np
import pandas as pd
# Load the "Data" sheet into a DataFrame
data_df = pd.read_excel('/mnt/data/MES2022Algo.xlsx', sheet_name='Data').iloc[2:] \
    .reset_index(drop=True)

# Drop NaN columns and rows
data_df = data_df.dropna(how='all', axis=1)
data_df = data_df.dropna(how='all', axis=0).reset_index(drop=True)

# Convert relevant columns to appropriate data types
data_df['Date'] = pd.to_datetime(data_df['Date'])
data_df[['Open', 'High', 'Low', 'Close']] = data_df[['Open', 'High', 'Low', 'Close']] \
    .astype(float)

# Initialize new columns
data_df['Direction'] = None
data_df['Decline'] = np.nan

# Calculate "Direction" based on Close prices
data_df['Direction'] = data_df['Close'].diff().apply(lambda x: 'Up' if x > 0 else \
                                                     ('Down' if x < 0 else 'Sideways'))

# Identify Buy, Sell, and Short Days (assuming the cycle starts at the first row, \
# i.e., index 0)
cycle_mapping = {0: 'Buy Day', 1: 'Sell Day', 2: 'Short Day'}
data_df['Cycle Day'] = [cycle_mapping[i % 3] for i in range(len(data_df))]

# Calculate "Decline" for Buy Days
for i in range(len(data_df)):
    if data_df.loc[i, 'Cycle Day'] == 'Buy Day' and i >= 3:  # Skip the first Buy Day
        data_df.loc[i, 'Decline'] = data_df.loc[i - 1, 'High'] - data_df.loc[i, 'Low']

# Show the DataFrame with new columns added
data_df.head(10)
