import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import pandas_market_calendars as mcal

pd.set_option('display.max_columns', None)


def apply_time_of_day_latency(data, mean_latency=5, std_latency=1, open_close_factor=2, midday_factor=0.5):
    latencies = np.random.normal(loc=mean_latency, scale=std_latency, size=len(data))
    nyse = mcal.get_calendar('NYSE')

    schedule = nyse.schedule.loc[data.index.date[0]:data.index.date[-1]]
    open_close_indices = []
    midday_indices = []

    for date, times in schedule.iterrows():
        market_open = times['market_open']
        market_close = times['market_close']
        open_period = (data.index >= market_open) & (data.index < market_open + pd.Timedelta(hours=1.5))
        open_close_indices.append(open_period)
        close_period = (data.index >= market_close - pd.Timedelta(hours=1))
        open_close_indices.append(close_period)

        midday_period = (data.index >= market_open + pd.Timedelta(hours=1.5)) & ( data.index < market_close - pd.Timedelta(hours=1))
        midday_indices.append(midday_period)

    open_close_indices = np.any(open_close_indices, axis=0)
    midday_indices = np.any(midday_indices, axis=0)
    latencies[open_close_indices] *= open_close_factor
    latencies[midday_indices] *= midday_factor
    simulated_timestamps = data.index + pd.to_timedelta(latencies, unit='s')

    for i in range(1, len(simulated_timestamps)):
        if simulated_timestamps[i] < simulated_timestamps[i - 1]:
            simulated_timestamps[i] = simulated_timestamps[i - 1] + pd.Timedelta(seconds=1)

    data['Simulated_Timestamp'] = simulated_timestamps
    return data, latencies


def detect_timestamp_crossover(data):
    crossover_indices = data['Simulated_Timestamp'] < data['Simulated_Timestamp'].shift(1)
    crossover_points = data[crossover_indices]
    return crossover_points

data = yf.download(
    'AAPL',  # Ticker symbol
    start='2024-08-21',  # Start date
    end='2024-08-23',  # End date (up to but not including this date)
    interval='1m'  # Interval set to one minute
)

latency_mean = 5
latency_std = 1 
open_close_factor = 2
midday_factor = 0.5

simulated_data_with_latency, latencies = apply_time_of_day_latency(data, latency_mean, latency_std, open_close_factor, midday_factor)
crossover_points = detect_timestamp_crossover(simulated_data_with_latency)
print(f"Number of Timestamp Crossovers Detected: {len(crossover_points)}")
print("Crossover Points Detected:")
print(crossover_points)

plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Original', marker='o')
plt.plot(simulated_data_with_latency['Simulated_Timestamp'], simulated_data_with_latency['Close'],
         label='Simulated Data with Time-of-Day Latency', linestyle='--', marker='x')
plt.legend()
plt.title('Simulated Data with Time-of-Day Gaussian Latency on Historical Data')
plt.show()

# Plot the distribution of latencies (in seconds)
plt.figure(figsize=(10, 6))
plt.hist(latencies, bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of Latencies Added')
plt.xlabel('Latency (seconds)')
plt.ylabel('Frequency')
plt.show()
