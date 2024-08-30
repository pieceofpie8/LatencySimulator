import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

pd.set_option('display.max_columns', None)

def apply_gaussian_latency(data, mean_latency=86400, std_latency=3600, volume_factor=2, volume_threshold_percentile=95):
    # Generate Gaussian-distributed latencies
    latencies = np.random.normal(loc=mean_latency, scale=std_latency, size=len(data))

    # Calculate the volume threshold for "high volume"
    volume_threshold = np.percentile(data['Volume'], volume_threshold_percentile)

    # Increase latency during high-volume periods
    high_volume_indices = data['Volume'] > volume_threshold
    latencies[high_volume_indices] *= volume_factor

    # Apply latency to timestamps
    simulated_timestamps = data.index + pd.to_timedelta(latencies, unit='s')

    data['Simulated_Timestamp'] = simulated_timestamps
    return data, latencies

def detect_timestamp_crossover(data):
    # Identify where simulated timestamps are earlier than the previous timestamp
    crossover_indices = data['Simulated_Timestamp'] < data['Simulated_Timestamp'].shift(1)
    crossover_points = data[crossover_indices]
    return crossover_points

# Fetch historical data
data = yf.download('AAPL', start='2022-01-01', end='2023-01-01')

# Apply Gaussian latency to the data with 24 hours latency
latency_mean = 86400  # Mean latency in seconds (24 hours)
latency_std = 3600    # Standard deviation of latency in seconds (1 hour)
volume_factor = 1.24     # Factor to increase latency during high-volume periods
volume_threshold_percentile = 75  # Top 25% volume considered "high volume"

simulated_data_with_latency, latencies = apply_gaussian_latency(
    data, latency_mean, latency_std, volume_factor, volume_threshold_percentile)

# Detect timestamp crossovers
crossover_points = detect_timestamp_crossover(simulated_data_with_latency)

# Print the number of crossover points
print(f"Number of Timestamp Crossovers Detected: {len(crossover_points)}")

# Print the crossover points, if any
print(crossover_points)

# Plot the original vs simulated data with latency
plt.figure(figsize=(14,7))
plt.plot(data.index, data['Close'], label='Original', marker='o')
plt.plot(simulated_data_with_latency['Simulated_Timestamp'], simulated_data_with_latency['Close'], label='Simulated Data with 24 Hours Latency', linestyle='--', marker='x')
plt.legend()
plt.title('Simulated Data with 24 Hours Gaussian Latency on Historical Data')
plt.show()

# Plot the distribution of latencies (in hours)
plt.figure(figsize=(10,6))
plt.hist(latencies / 3600, bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of Latencies Added')
plt.xlabel('Latency (hours)')
plt.ylabel('Frequency')
plt.show()
