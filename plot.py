import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from sim_search import find_similar


# Read the data
data1 = pd.read_csv('data/actual_temperature.csv', parse_dates=['date'])
data2 = pd.read_csv('data/wanted_temperature.csv', parse_dates=['date'])
weather_data = pd.read_csv('data/weather_data.csv')


# Simplify the "date" section in a dataframe for better readability
def add_simplified_time(df, length, start_time):
    st = start_time
    df['simple_time'] = [st + timedelta(hours = i) for i in range (len(df))]
    return df


# Apply simplified time for both datasets
start_time = datetime(2025, 2, 22, 12, 0)
length = len(data1)
data1 = add_simplified_time(data1, length, start_time)
data2 = add_simplified_time(data2, length, start_time)
weather_data = add_simplified_time(weather_data, length, start_time)
weather_data['outdoor_temp'] = (weather_data['Maksimumstemperatur']+weather_data['Minimumstemperatur'])/2


# Set the 'simple_time' as the index for plotting
data1.set_index('simple_time', inplace=True)
data2.set_index('simple_time', inplace=True)
weather_data.set_index('simple_time', inplace=True)

# Create dataframe usd for similarity search
sim_search_data = pd.DataFrame()
sim_search_data = add_simplified_time(sim_search_data, length, start_time)
sim_search_data.set_index('simple_time', inplace=True)

# Adding avialable data to the dataframe, which can be used for similarity search
sim_search_data['measured_temp'] = data1['value']

# Columns to be used for similarity search
sim_search_cols = ['measured_temp']

# Check if all columns needed for similarity search exist
for col in sim_search_cols:
    if col not in sim_search_data.columns:
        print(f"Error: Column '{col}' not found in dataframe")
        print(f"Available columns: {list(sim_search_data.columns)}")
        exit(1)

# Parameters for similarity search
query = datetime(2025, 3, 10, 18, 0)
w_size = 3

result, succeed = find_similar(pd_data= sim_search_data, columns= sim_search_cols, 
                               top_k= 5, window_size= w_size, 
                               query_start= query
                               )

if not succeed:
    exit(1)

print(pd.DataFrame(result))

# Create the main plot
fig, ax = plt.subplots(figsize=(14, 8))
line1 = ax.plot(data1.index, data1['value'], label='Actual Temp', color='blue', alpha=0.7, zorder=1)[0]
line2 = ax.plot(data2.index, data2['value'], label='Wanted Temp', color='orange', linestyle='--', alpha=0.7, zorder=1)[0]
line3 = ax.plot(weather_data.index, weather_data['outdoor_temp'], label='Outdoor Temp', color='green', linestyle=':', alpha=0.5, zorder=1)[0]

# Highlight the query window
query_end = query + timedelta(hours=w_size-1)  # Adjust to correct endpoint
ax.axvspan(query, query_end, color='yellow', alpha=0.3, label='Query Window', zorder=0)

# Plot the top matches
colors = ['red', 'purple', 'brown', 'teal', 'magenta']
for i, match in enumerate(result[:min(5, len(result))]):
    start_time = match['start_time']
    end_time = start_time + timedelta(hours=w_size-1)  # Adjust to correct endpoint
    
    # Get the actual data points for this window
    match_window_data = data1.loc[start_time:end_time]
    
    # Plot with unique colors for each match
    match_color = colors[i % len(colors)]
    
    # Plot the match with connecting lines
    ax.plot(match_window_data.index, match_window_data['value'], 
            color=match_color, linewidth=2, marker='o', markersize=5, alpha=0.8, zorder=2,
            label=f"Match {i+1}")
    
    # Lighter highlight for the match region
    ax.axvspan(start_time, end_time, color=match_color, alpha=0.1, zorder=0)

# Fill between actual and wanted temperature
ax.fill_between(data1.index, data1['value'], data2['value'], 
                where=(data1['value'] > data2['value']), color='red', alpha=0.1, label='Overshoot', zorder=0)
ax.fill_between(data1.index, data1['value'], data2['value'], 
                where=(data1['value'] < data2['value']), color='blue', alpha=0.1, label='Undershoot', zorder=0)

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
        fig.canvas.draw_idle()

# Connect the update function to zoom events
fig.canvas.mpl_connect('draw_event', update_ticks)

# Initial tick setup
update_ticks()

plt.title('Temperature Data with Query Window and Similar Matches')
plt.xlabel('Date and Time')
plt.ylabel('Temperature (°C)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()
plt.show()


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