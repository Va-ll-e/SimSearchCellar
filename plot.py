import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Read the data
data1 = pd.read_csv('data/actual_temperature.csv', parse_dates=['date'])
data2 = pd.read_csv('data/wanted_temperature.csv', parse_dates=['date'])


# Simplify the "date" section in a dataframe for better readability
def add_simplified_time(df, start_time):
    st = start_time
    df['simple_time'] = [st + timedelta(hours = i) for i in range (len(df))]
    df['simple_time_string'] = df['simple_time'].dt.strftime('%Y-%b-%d -- %H:%M')
    return df


# Apply simplified time for both datasets
start_time = datetime(2025, 2, 21, 11, 0)
data1 = add_simplified_time(data1, start_time)
data2 = add_simplified_time(data2, start_time)


# Set the 'simple_time' as the index for plotting
data1.set_index('simple_time', inplace=True)
data2.set_index('simple_time', inplace=True)


data1['diff'] = data1['value'].diff()
abs_threshold = 0.6
drops = data1[data1['diff'] <= (-abs_threshold)]
spikes = data1[data1['diff'] >= abs_threshold]

plt.figure(figsize=(12,6))
plt.plot(data1.index, data1['value'], label='Temp', color='blue')
plt.scatter(drops.index, drops['value'], color='red', label='Drop', zorder=3)
plt.scatter(spikes.index, spikes['value'], color='green', label='Spike', zorder=3)


plt.gca().set_xticks(data1.index[::24])
plt.gca().set_xticklabels(data1['simple_time_string'][::24], rotation=90, ha='right')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


print(f"Drops: {drops['diff'].count()}")
print(f"Spikes: {spikes['diff'].count()}")
