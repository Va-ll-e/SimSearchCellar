import pandas as pd
import numpy as np
import faiss
from datetime import datetime, timedelta


# Read the data
actual_data = pd.read_csv('data/actual_temperature.csv', parse_dates=['date'])
set_data = pd.read_csv('data/wanted_temperature.csv', parse_dates=['date'])
weather_data = pd.read_csv('data/weather_data.csv')


# Simplify the "date" section in a dataframe for better readability
def add_simplified_time(df, start_time):
    st = start_time
    df['simple_time'] = [st + timedelta(hours = i) for i in range (len(df))]
    df['simple_time_string'] = df['simple_time'].dt.strftime('%Y-%b-%d -- %H:%M')
    return df


# Define the create_window_embeddings function
def create_window_embeddings(df, window_size, features):
    """
    Create embeddings from sliding windows of the specified features.
    
    Args:
        df: DataFrame containing the data
        window_size: Size of the sliding window
        features: List of column names to include in embeddings
        
    Returns:
        embeddings: numpy array of embeddings
        timestamps: list of timestamps corresponding to the start of each window
    """
    embeddings = []
    timestamps = []
    
    for i in range(len(df) - window_size + 1):
        # Get data for the current window
        window_data = df.iloc[i:i+window_size][features].values.flatten()
        embeddings.append(window_data)
        timestamps.append(df.index[i])
    
    return np.array(embeddings, dtype='float32'), timestamps


# Apply simplified time for both datasets
start_time = datetime(2025, 2, 22, 12, 0)
actual_data = add_simplified_time(actual_data, start_time)
set_data = add_simplified_time(set_data, start_time)
weather_data = add_simplified_time(weather_data, start_time)
weather_data['outdoor_temp'] = (weather_data['Maksimumstemperatur']+weather_data['Minimumstemperatur'])/2


# Set the 'simple_time' as the index for plotting
actual_data.set_index('simple_time', inplace=True)
set_data.set_index('simple_time', inplace=True)
weather_data.set_index('simple_time', inplace=True)


actual_data['diff'] = actual_data['value'].diff()
abs_threshold = 0.7
drops = actual_data[actual_data['diff'] <= (-abs_threshold)]
spikes = actual_data[actual_data['diff'] >= abs_threshold]


# Fix the warning by using ffill() instead of fillna(method='ffill')
data_combined = actual_data.join(weather_data['outdoor_temp'], how='left').ffill()


# Create embeddings
window_size = 5
features = ['value']
embeddings, timestamps = create_window_embeddings(data_combined, window_size, features)
print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")


# FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"FAISS index contains {index.ntotal} vectors")

# Query
query_start = datetime(2025, 3, 10, 17, 0)
query_end = datetime(2025, 3, 10, 21, 0)


# Flag to track if search was successful
search_succeeded = False


# Create query embedding with the same window_size as the indexed embeddings
# Check if the query window has enough data points
if query_end - query_start >= timedelta(hours=window_size-1):
    # Extract the query window data
    query_df = data_combined.loc[query_start:query_end]
    # Use the same window_size and features as the index
    if len(query_df) >= window_size:
        query_data = query_df[features].iloc[:window_size].values.flatten()
        query_embedding = np.array([query_data], dtype='float32')
        
        # Search with more results initially to allow for filtering
        initial_k = 15  # Get more results initially
        min_hours_between_results = (abs(query_start-query_end).total_seconds()/3600)-1  # Minimum hours between results to avoid overlaps
        distances, indices = index.search(query_embedding, initial_k)
        
        # Filter to remove overlapping results
        filtered_indices = []
        filtered_distances = []
        

        # Stop once we have x non-overlapping results
        x = 5
        for i, idx in enumerate(indices[0]):
            # Skip the result if it's the query itself (exact match with zero distance)
            if abs(distances[0][i]) < 1e-5 and data_combined.index[idx] == query_start:
                continue
                
            # Check if this result overlaps with any previously selected result
            overlaps = False
            for prev_idx in filtered_indices:
                time_diff = abs((data_combined.index[idx] - data_combined.index[prev_idx]).total_seconds() / 3600)
                if time_diff < min_hours_between_results:
                    overlaps = True
                    break
                    
            if not overlaps:
                filtered_indices.append(idx)
                filtered_distances.append(distances[0][i])
                
            if len(filtered_indices) >= x:
                break
        
        search_succeeded = True
        
        # Results
        print(f"\nTop 5 Similar Windows to {query_start}:")
        for i, (dist, idx) in enumerate(zip(filtered_distances, filtered_indices)):
            start_time = timestamps[idx]
            window_values = data_combined['value'].iloc[idx:idx + window_size].values
            print(f"{i+1}. Start: {data_combined.loc[start_time, 'simple_time_string']}, "
                f"Temps: {window_values}, Distance: {dist:.2f}")
    else:
        print(f"Not enough data points in the query window. Found {len(query_df)}, need {window_size}")
else:
    print(f"{'-'*50}\nQuery window is too small. It should span at least {window_size} hours\n{'-'*50}")