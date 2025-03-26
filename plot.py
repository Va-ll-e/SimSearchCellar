import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from sim_search import find_similar

# Read data files
data1 = pd.read_csv('data/actual_temperature.csv', parse_dates=['date'])
data2 = pd.read_csv('data/wanted_temperature.csv', parse_dates=['date'])
weather_data = pd.read_csv('data/weather_data.csv')

# Add simplified time index to dataframes
def prepare_dataframe(df, start_time):
    df['simple_time'] = [start_time + timedelta(hours=i) for i in range(len(df))]
    df.set_index('simple_time', inplace=True)
    return df

# Prepare all dataframes
start_time = datetime(2025, 2, 22, 12, 0)
data1 = prepare_dataframe(data1, start_time)
data2 = prepare_dataframe(data2, start_time)
weather_data = prepare_dataframe(weather_data, start_time)
weather_data['outdoor_temp'] = (weather_data['Maksimumstemperatur'] + weather_data['Minimumstemperatur']) / 2

# Setup similarity search dataframe
sim_search_data = pd.DataFrame({'measured_temp': data1['value']}, index=data1.index)
sim_search_cols = ['measured_temp']

# Run similarity search
query = datetime(2025, 3, 10, 17, 0)
w_size = 6
result, succeed = find_similar(
    pd_data=sim_search_data, 
    columns=sim_search_cols,
    top_k=5, 
    window_size=w_size,
    query_start=query
)

if not succeed:
    print("Similarity search failed")
    exit(1)

print(pd.DataFrame(result))

# Create plot
fig, ax = plt.subplots(figsize=(14, 8))

# Plot main data
ax.plot(data1.index, data1['value'], label='Actual Temp', color='blue', alpha=0.7)
ax.plot(data2.index, data2['value'], label='Wanted Temp', color='orange', linestyle='--', alpha=0.7)
#ax.plot(weather_data.index, weather_data['outdoor_temp'], label='Outdoor Temp', color='green', linestyle=':', alpha=0.5)

# Highlight query window
query_end = query + timedelta(hours=w_size-1)
ax.axvspan(query, query_end, color='yellow', alpha=0.3, label='Query Window')

# Plot top matches
colors = ['red', 'magenta']
for i, match in enumerate(result[:min(5, len(result))]):
    start_time = match['start_time']
    end_time = start_time + timedelta(hours=w_size-1)
    match_data = data1.loc[start_time:end_time]
    
    # Plot match with unique color
    match_color = colors[i % len(colors)]
    ax.plot(match_data.index, match_data['value'], 
            color=match_color, linewidth=2, marker='o', markersize=5, alpha=0.8,
            label=f"Match {i+1}")
    ax.axvspan(start_time, end_time, color=match_color, alpha=0.1)

# Fill areas between actual and wanted temperatures
ax.fill_between(data1.index, data1['value'], data2['value'], 
                where=(data1['value'] > data2['value']), color='red', alpha=0.1, label='Overshoot')
ax.fill_between(data1.index, data1['value'], data2['value'], 
                where=(data1['value'] < data2['value']), color='blue', alpha=0.1, label='Undershoot')

# Dynamic tick adjustment function
def update_ticks(event=None):
    xlim = ax.get_xlim()
    start_idx = max(0, int(xlim[0]))
    end_idx = min(len(data1), int(xlim[1]))
    visible_range = end_idx - start_idx
    
    interval = 6 if visible_range > 150 else 3 if visible_range > 50 else 1
    
    tick_positions = [i for i in range(start_idx, end_idx, interval) if i < len(data1.index)]
    if tick_positions:
        ax.set_xticks([data1.index[i] for i in tick_positions])
        fig.canvas.draw_idle()

# Connect update function to zoom events
fig.canvas.mpl_connect('draw_event', update_ticks)
update_ticks()  # Initial setup

# Finalize plot
plt.title('Temperature Data with Query Window and Similar Matches')
plt.xlabel('Date and Time')
plt.ylabel('Temperature (°C)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate and display temperature deviation statistics
data1['deviation'] = data1['value'] - data2['value']
threshold = 1.0

print(f"\nAverage Deviation: {data1['deviation'].mean():.2f}°C")
print(f"Max Overshoot: {data1['deviation'].max():.2f}°C")
print(f"Max Undershoot: {data1['deviation'].min():.2f}°C")
print(f"Time at exact temperature: {(data1['deviation'] == 0.0).sum() / len(data1) * 100:.2f}%")
print(f"Time within {threshold}°C threshold: {(abs(data1['deviation']) < threshold).sum() / len(data1) * 100:.2f}%")