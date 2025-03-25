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
start_time = datetime(2025, 2, 22, 12, 0)
data1 = add_simplified_time(data1, start_time)
data2 = add_simplified_time(data2, start_time)


# Set the 'simple_time' as the index for plotting
data1.set_index('simple_time', inplace=True)
data2.set_index('simple_time', inplace=True)


data1['diff'] = data1['value'].diff()
abs_threshold = 0.7
drops = data1[data1['diff'] <= (-abs_threshold)]
spikes = data1[data1['diff'] >= abs_threshold]


plt.figure(figsize=(12, 6))
plt.plot(data1.index, data1['value'], label='Actual Temp', color='blue')
plt.plot(data2.index, data2['value'], label='Wanted Temp', color='orange', linestyle='--')
plt.fill_between(data1.index, data1['value'], data2['value'], 
                 where=(data1['value'] > data2['value']), color='red', alpha=0.3, label='Overshoot')
plt.fill_between(data1.index, data1['value'], data2['value'], 
                 where=(data1['value'] < data2['value']), color='cyan', alpha=0.3, label='Undershoot')
plt.scatter(drops.index, drops['value'], color='red', label='Drop', zorder=3)
plt.scatter(spikes.index, spikes['value'], color='green', label='Spike', zorder=3)

plt.gca().set_xticks(data1.index[::24])
plt.gca().set_xticklabels(data1['simple_time_string'][::24], rotation=90, ha='right')

plt.title('Actual vs Wanted Temperature with Deviations')
plt.xlabel('Date and Time')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Summary stats for drops and spikes
print("\n\nDrop Details:")
print(drops[['simple_time_string', 'value', 'diff']].to_string(index=False))
print("\nSpike Details:")
print(spikes[['simple_time_string', 'value', 'diff']].to_string(index=False))

# Calculate deviation from wanted temperature
data1['deviation'] = data1['value'] - data2['value']
avg_deviation = data1['deviation'].mean()
max_overshoot = data1['deviation'].max()
max_undershoot = data1['deviation'].min()
correct_temp = (data1['deviation'] == 0.0).sum() / len(data1)
# Acceptable threshold in degrees
threshold = 1.0
accept_diff = (abs(data1['deviation']) < threshold).sum() / len(data1)

print(f"\nAverage Deviation from Wanted Temp: {avg_deviation:.2f}°C")
print(f"Max Overshoot: {max_overshoot:.2f}°C")
print(f"Max Undershoot: {max_undershoot:.2f}°C")
print(f"Percentage of time correct temperature is held: {correct_temp*100:.2f}%")
print(f"Percentage of time temperature is held within {threshold}-degree threshold: {accept_diff*100:.2f}%")
