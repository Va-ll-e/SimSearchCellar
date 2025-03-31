import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from sim_search import find_similar


# Read data files
measured_temp = pd.read_csv('data/actual_temperature.csv', parse_dates=['date'])
set_temp = pd.read_csv('data/wanted_temperature.csv', parse_dates=['date'])
weather_data = pd.read_csv('data/weather_data.csv')


# Add simplified time index to dataframes
def prepare_dataframe(df, start_time):
    df['simple_time'] = [start_time + timedelta(hours=i) for i in range(len(df))]
    df.set_index('simple_time', inplace=True)
    return df


# Prepare all dataframes
start_time = datetime(2025, 2, 22, 12, 0)
measured_temp = prepare_dataframe(measured_temp, start_time)
set_temp = prepare_dataframe(set_temp, start_time)
weather_data = prepare_dataframe(weather_data, start_time)
weather_data['outdoor_temp'] = (weather_data['Maksimumstemperatur'] + weather_data['Minimumstemperatur']) / 2


# Setup similarity search dataframe
sim_search_data = pd.DataFrame({'measured_temp': measured_temp['value']}, index=measured_temp.index)
sim_search_data['set_temp'] = set_temp['value']
sim_search_data['out_temp'] = weather_data['outdoor_temp']


# Can include 'measured_temp', 'set_temp' and 'out_temp'
# Can also include others if they are added as columns sim_search_data
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
ax.plot(measured_temp.index, measured_temp['value'], label='Actual Temp', color='blue', alpha=0.7)
ax.plot(set_temp.index, set_temp['value'], label='Set Temp', color='orange', linestyle='--', alpha=0.7)
ax.plot(weather_data.index, weather_data['outdoor_temp'], label='Outdoor Temp', color='green', linestyle=':', alpha=0.5)


# Highlight query window
query_end = query + timedelta(hours=w_size-1)
ax.axvspan(query, query_end, color='yellow', alpha=0.3, label='Query Window')


# Plot top matches
colors = ['red', 'magenta']
for i, match in enumerate(result[:min(5, len(result))]):
    start_time = match['start_time']
    end_time = start_time + timedelta(hours=w_size-1)
    match_data = measured_temp.loc[start_time:end_time]
    

    # Plot match with unique color
    match_color = colors[i % len(colors)]
    ax.plot(match_data.index, match_data['value'], 
            color=match_color, linewidth=2, marker='o', markersize=5, alpha=0.8,
            label=f"Match {i+1}")
    ax.axvspan(start_time, end_time, color=match_color, alpha=0.1)


# Fill areas between actual and Set temperatures
ax.fill_between(measured_temp.index, measured_temp['value'], set_temp['value'], 
                where=(measured_temp['value'] > set_temp['value']), color='red', alpha=0.1, label='Overshoot')
ax.fill_between(measured_temp.index, measured_temp['value'], set_temp['value'], 
                where=(measured_temp['value'] < set_temp['value']), color='blue', alpha=0.1, label='Undershoot')


# Dynamic tick adjustment function
def update_ticks(event=None):
    xlim = ax.get_xlim()
    start_idx = max(0, int(xlim[0]))
    end_idx = min(len(measured_temp), int(xlim[1]))
    visible_range = end_idx - start_idx
    
    interval = 6 if visible_range > 150 else 3 if visible_range > 50 else 1
    
    tick_positions = [i for i in range(start_idx, end_idx, interval) if i < len(measured_temp.index)]
    if tick_positions:
        ax.set_xticks([measured_temp.index[i] for i in tick_positions])
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
measured_temp['deviation'] = measured_temp['value'] - set_temp['value']
threshold = 1.0

print(f"\nAverage Deviation: {measured_temp['deviation'].mean():.2f}°C")
print(f"Max Overshoot: {measured_temp['deviation'].max():.2f}°C")
print(f"Max Undershoot: {measured_temp['deviation'].min():.2f}°C")
print(f"Time at exact temperature: {(measured_temp['deviation'] == 0.0).sum() / len(measured_temp) * 100:.2f}%")
print(f"Time within {threshold}°C threshold: {(abs(measured_temp['deviation']) < threshold).sum() / len(measured_temp) * 100:.2f}%")