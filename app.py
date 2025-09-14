# This is the start of MarketEnvelopePro 
import numpy as np
import pandas as pd
# Load the "Data" sheet into a DataFrame
data_df = pd.read_excel('/mnt/data/mesPriceHistory.xlsx', sheet_name='Data').iloc[2:] \
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



# Add Volume and Range columns
data_df['Volume'] = np.random.randint(1000, 5000, size=len(data_df))  # Using random \
        #values as placeholders for Volume
data_df['Volume_AVG'] = data_df['Volume'].expanding().mean()

# Add Range to data_df and calculate its average
data_df['Range'] = data_df['High'] - data_df['Low']
data_df['Range_AVG'] = data_df['Range'].expanding().mean()

# Update Forecast data with Volume-AVG and Range-AVG
forecast_data.update({
    'Volume Average': data_df['Volume_AVG'].iloc[-1],
    'Range Average': data_df['Range_AVG'].iloc[-1]
})



# Calculate 'BH' and 'BU' for Buy Days
for i in range(1, len(data_df)):  # Start from index 1 to avoid negative indexing
    cycle_day = data_df.loc[i, 'Cycle Day']
    
    if cycle_day == 'Buy Day':
        # Calculate 'BH' on Buy Days
        short_day_high = data_df.loc[i - 1, 'High'] if data_df.loc[i - 1, 'Cycle Day'] == 'Short Day' else np.nan
        if short_day_high is not np.nan:
            bh_value = data_df.loc[i, 'High'] - short_day_high
            if bh_value > 0:
                data_df.loc[i, 'BH'] = bh_value
            else:
                data_df.loc[i, 'BH'] = False  # Set to False when the condition is not met
        
        # Calculate 'BU' on Buy Days
        short_day_low = data_df.loc[i - 1, 'Low'] if data_df.loc[i - 1, 'Cycle Day'] == 'Short Day' else np.nan
        if short_day_low is not np.nan:
            bu_value = short_day_low - data_df.loc[i, 'Low']
            if bu_value > 0:
                data_df.loc[i, 'BU'] = bu_value
            elif data_df.loc[i, 'Decline'] == 0:
                data_df.loc[i, 'BU'] = 0  # Set to 0 when Decline is zero on a Buy Day

# Initialize flags for BU zero decline
bu_zero_decline_dates = []

# Check for zero decline in BU on Buy Days
bu_zero_decline_dates = data_df[(data_df['Cycle Day'] == 'Buy Day') & (data_df['BU'] == 0)]['Date'].tolist()

# Calculate averages for BH and BU
data_df['BH_AVG'] = data_df[data_df['Cycle Day'] == 'Buy Day']['BH'].apply(pd.to_numeric, errors='coerce').expanding().mean()
data_df['BU_AVG'] = data_df[data_df['Cycle Day'] == 'Buy Day']['BU'].apply(pd.to_numeric, errors='coerce').expanding().mean()

# Update Forecast data with BH_AVG and BU_AVG
forecast_data.update({
    'BH Average': data_df['BH_AVG'].iloc[-1] if not data_df['BH_AVG'].isna().all() else np.nan,
    'BU Average': data_df['BU_AVG'].iloc[-1] if not data_df['BU_AVG'].isna().all() else np.nan,
    'BU Zero Decline Dates': bu_zero_decline_dates
})
# Initialize new columns for 'Buy Violation' and 'Buy Violation Points'

data_df['Buy Violation'] = False

data_df['Buy Violation Points'] = np.nan



# Calculate 'Buy Violation' and 'Buy Violation Points'

for i in range(1, len(data_df)):  # Start from index 1 to avoid negative indexing

    current_day_low = data_df.loc[i, 'Low']

    prior_day_low = data_df.loc[i - 1, 'Low']

    

    if current_day_low < prior_day_low:

        data_df.loc[i, 'Buy Violation'] = True

        data_df.loc[i, 'Buy Violation Points'] = prior_day_low - current_day_low



# Initialize list to flag Buy Violations for the Forecast sheet

buy_violation_dates = []

buy_violation_points = []



# Check for Buy Violation on all days and note the point difference

buy_violation_data = data_df[data_df['Buy Violation'] == True][['Date', 'Buy Violation Points']]

buy_violation_dates = buy_violation_data['Date'].tolist()

buy_violation_points = buy_violation_data['Buy Violation Points'].tolist()



# Update Forecast data with Buy Violation information

forecast_data.update({

    'Buy Violation Dates': buy_violation_dates,

    'Buy Violation Points': buy_violation_points

})
# Calculate 'Trend Reaction', 'TrendReaction Buy', and 'TrendReaction Sell'

data_df['Trend Reaction'] = (data_df['High'] + data_df['Low'] + data_df['Close']) / 3

data_df['TrendReaction Buy'] = 2 * (data_df['Trend Reaction'] - data_df['High'])

data_df['TrendReaction Sell'] = 2 * (data_df['Trend Reaction'] - data_df['Low'])

# Calculate 'Trend MoMo Indicator'
data_df['Trend MoMo Indicator'] = data_df['Close'].diff(periods=2)

# Initialize 'Trend MoMo Direction'
data_df['Trend MoMo Direction'] = np.nan

# Calculate 'Trend MoMo Direction'
for i in range(4, len(data_df)):  # Start from index 4 to avoid negative indexing and to have 2 prior values
    current_momo = data_df.loc[i, 'Trend MoMo Indicator']
    two_prior_momo = data_df.loc[i - 2, 'Trend MoMo Indicator']
    four_prior_momo = data_df.loc[i - 4, 'Trend MoMo Indicator']
    
    if current_momo > two_prior_momo and current_momo > four_prior_momo:
        data_df.loc[i, 'Trend MoMo Direction'] = 'Up'
    elif current_momo < two_prior_momo and current_momo < four_prior_momo:
        data_df.loc[i, 'Trend MoMo Direction'] = 'Down'
    else:
        data_df.loc[i, 'Trend MoMo Direction'] = 'Sideways'

# Initialize list to flag Trend MoMo Direction for the Forecast sheet
trend_momo_direction_latest = data_df['Trend MoMo Direction'].iloc[-1]

# Update Forecast data with Trend MoMo Direction
forecast_data.update({
    'Latest Trend MoMo Direction': trend_momo_direction_latest
})
# Calculate 'LSS DECLINE', 'LSS Rally', and 'LSS BuyHigh'

data_df['LSS DECLINE'] = data_df['High'].shift(1) - data_df['Low']

data_df['LSS Rally'] = data_df['High'] - data_df['Low'].shift(1)

data_df['LSS BuyHigh'] = data_df['High'] - data_df['High'].shift(1)



# Calculate their 3-day ATR Averages

data_df['LSS Decline 3Day ATR AVG'] = data_df['LSS DECLINE'].rolling(window=3).mean()

data_df['LSS Rally 3Day ATR AVG'] = data_df['LSS Rally'].rolling(window=3).mean()

data_df['LSS Buy High 3Day ATR AVG'] = data_df['LSS BuyHigh'].rolling(window=3).mean()

# Calculate 'LSS BuyUnder'

data_df['LSS BuyUnder'] = data_df['Low'] - data_df['Low'].shift(1)



# Calculate 'Lss BuyUnder 3Day ATR AVG'

data_df['Lss BuyUnder 3Day ATR AVG'] = data_df['LSS BuyUnder'].rolling(window=3).mean()



# Calculate 'LSS_Decline_AVG' as the average of all 'LSS DECLINE'

data_df['LSS_Decline_AVG'] = data_df['LSS DECLINE'].expanding().mean()



# Calculate 'LSS_Decline_3Day_ATR' as the average of 'LSS Decline 3Day ATR AVG'

data_df['LSS_Decline_3Day_ATR'] = data_df['LSS Decline 3Day ATR AVG'].expanding().mean()
# Calculate 'LSS_Rally_AVG', 'LSS_Rally_3day_ATR', 'Lss_BuyHigh_AVG', 'LSS_BuyHigh_3day_AVG', 'LSS_BU_AVG', and 'LSS_BuyUnder_3Day_AVG'

data_df['LSS_Rally_AVG'] = data_df['LSS Rally'].expanding().mean()

data_df['LSS_Rally_3day_ATR'] = data_df['LSS Rally 3Day ATR AVG'].expanding().mean()

data_df['Lss_BuyHigh_AVG'] = data_df['LSS BuyHigh'].expanding().mean()

data_df['LSS_BuyHigh_3day_AVG'] = data_df['LSS Buy High 3Day ATR AVG'].expanding().mean()

data_df['LSS_BU_AVG'] = data_df['BU'].expanding().mean()  # Assuming 'LSS_BU_AVG' is the average of the 'BU' column

data_df['LSS_BuyUnder_3Day_AVG'] = data_df['Lss BuyUnder 3Day ATR AVG'].expanding().mean()



# Update Forecast data with the most current date's Open, High, Low, and Close

forecast_data.update({

    'Today Open': data_df['Open'].iloc[-1],

    'Today High': data_df['High'].iloc[-1],

    'Today Low': data_df['Low'].iloc[-1],

    'Today Close': data_df['Close'].iloc[-1]

})
# Update Forecast data with the "Yesterday's" Open, High, Low, and Close
forecast_data.update({
    'Yesterday Open': data_df['Open'].iloc[-2] if len(data_df) > 1 else np.nan,
    'Yesterday High': data_df['High'].iloc[-2] if len(data_df) > 1 else np.nan,
    'Yesterday Low': data_df['Low'].iloc[-2] if len(data_df) > 1 else np.nan,
    'Yesterday Close': data_df['Close'].iloc[-2] if len(data_df) > 1 else np.nan
})


import matplotlib.pyplot as plt

# Sample forecast data for visualization; To be replaced with actual forecast data
sample_forecast_data = {
    'Zero Decline Dates': [],
    'Decline Average': 2.5,
    'Rally Average': 3.1,
    'Volume Average': 1500,
    'Range Average': 5.2,
    'BH Average': 1.2,
    'BU Average': 1.3,
    'Latest Trend MoMo Direction': 'Sideways',
    'Today Open': 4000,
    'Today High': 4025,
    'Today Low': 3980,
    'Today Close': 4010,
    'Yesterday Open': 3980,
    'Yesterday High': 4015,
    'Yesterday Low': 3975,
    'Yesterday Close': 4000
}

# Visualizing some of the key Forecast metrics
labels = list(sample_forecast_data.keys())
values = list(sample_forecast_data.values())

# Selecting only the numeric values for bar chart
numeric_labels = [label for label, value in zip(labels, values) if isinstance(value, (int, float))]
numeric_values = [value for value in values if isinstance(value, (int, float))]

plt.figure(figsize=(15, 6))
plt.barh(numeric_labels, numeric_values, color='skyblue')
plt.xlabel('Values')
plt.title('Forecast Key Metrics')
plt.show()



#CandleStick Chart
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc

# Generate sample data for visualization; this would be replaced by your actual data
sample_data = data_df.tail(10).copy()
sample_data['Date'] = mdates.date2num(sample_data.index.to_pydatetime())

# Prepare data in the format: Date, Open, High, Low, Close
plot_data = sample_data[['Date', 'Open', 'High', 'Low', 'Close']].values

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(12, 6))

# Configure x-axis
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

# Configure grid and labels
ax.grid(True)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Candlestick Chart of Daily Closing Prices')

# Create the candlestick chart
candlestick_ohlc(ax, plot_data, width=0.6, colorup='g', colordown='r')

plt.show()
