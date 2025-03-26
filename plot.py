import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.dates as mdates
from datetime import datetime, timedelta


# Read the data
data1 = pd.read_csv('data/actual_temperature.csv', parse_dates=['date'])
data2 = pd.read_csv('data/wanted_temperature.csv', parse_dates=['date'])
weather_data = pd.read_csv('data/weather_data.csv')


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
weather_data = add_simplified_time(weather_data, start_time)
weather_data['outdoor_temp'] = (weather_data['Maksimumstemperatur']+weather_data['Minimumstemperatur'])/2

# Set the 'simple_time' as the index for plotting
data1.set_index('simple_time', inplace=True)
data2.set_index('simple_time', inplace=True)
weather_data.set_index('simple_time', inplace=True)


data1['diff'] = data1['value'].diff()
abs_threshold = 0.7
drops = data1[data1['diff'] <= (-abs_threshold)]
spikes = data1[data1['diff'] >= abs_threshold]


fig, ax = plt.subplots(figsize=(12, 6))
line1 = ax.plot(data1.index, data1['value'], label='Actual Temp', color='blue')[0]
line2 = ax.plot(data2.index, data2['value'], label='Wanted Temp', color='orange', linestyle='--')[0]
line3 = ax.plot(weather_data.index, weather_data['outdoor_temp'], label='Outdoor Temp', color='green', linestyle=':')[0]

ax.fill_between(data1.index, data1['value'], data2['value'], 
                where=(data1['value'] > data2['value']), color='red', alpha=0.3, label='Overshoot')
ax.fill_between(data1.index, data1['value'], data2['value'], 
                where=(data1['value'] < data2['value']), color='cyan', alpha=0.3, label='Undershoot')
                
scatter1 = ax.scatter(drops.index, drops['value'], color='red', label='Drop', zorder=3)
scatter2 = ax.scatter(spikes.index, spikes['value'], color='green', label='Spike', zorder=3)


def update_ticks(event=None):
    # Get current x-axis limits
    xlim = ax.get_xlim()
    start_idx = max(0, int(xlim[0]))
    end_idx = min(len(data1), int(xlim[1]))
    
    # Calculate visible timespan
    visible_range = end_idx - start_idx
    
    # Dynamically adjust tick interval based on zoom level
    if visible_range > 150:  # Zoomed out - fewer ticks
        interval = max(6, visible_range // 20)
    elif visible_range > 50:  # Medium zoom
        interval = 3
    else:  # Zoomed in - more ticks
        interval = 1
    
    visible_indices = range(start_idx, end_idx)
    tick_positions = [i for i in visible_indices[::interval] if i < len(data1.index)]
    
    if tick_positions:
        ax.set_xticks([data1.index[i] for i in tick_positions])
        ax.set_xticklabels([data1['simple_time_string'].iloc[i] for i in tick_positions], 
                          rotation=90, ha='right')
        fig.canvas.draw_idle()

# Connect the update function to zoom events
fig.canvas.mpl_connect('draw_event', update_ticks)

# Initial tick setup
update_ticks()

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