def find_similar(pd_data, columns, top_k=5, window_size=3, query_start=None, query_stop=None):
    """
    Find similar windows of data based on the given query period.
    
    Args:
        pd_data: DataFrame containing the data with datetime index
        columns: List of column names to include in embeddings
        top_k: Number of similar windows to return (default: 5)
        window_size: Size of the sliding window (default: 3)
        query_start: Start datetime for the query window
        query_stop: End datetime for the query window
        
    Returns:
        results: List of dictionaries containing similar windows information
        search_succeeded: Boolean indicating if the search was successful
    """
    import numpy as np
    import faiss
    from datetime import timedelta
    
    # Create embeddings function
    def create_window_embeddings(df, window_size, features):
        """
        Create embeddings from sliding windows of the specified features.
        """
        embeddings = []
        timestamps = []
        
        for i in range(len(df) - window_size + 1):
            window_data = df.iloc[i:i+window_size][features].values.flatten()
            embeddings.append(window_data)
            timestamps.append(df.index[i])
        
        return np.array(embeddings, dtype='float32'), timestamps
    
    # Create embeddings
    embeddings, timestamps = create_window_embeddings(pd_data, window_size, columns)
    
    # FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Flag to track if search was successful
    search_succeeded = False
    results = []
    
    # Check if query window is large enough
    if query_stop - query_start >= timedelta(hours=window_size-1):
        # Extract the query window data
        query_df = pd_data.loc[query_start:query_stop]
        
        # Check if enough data points
        if len(query_df) >= window_size:
            query_data = query_df[columns].iloc[:window_size].values.flatten()
            query_embedding = np.array([query_data], dtype='float32')
            
            # Search with more results initially to allow for filtering
            initial_k = top_k * 3  # Get more results initially
            min_hours_between_results = (abs(query_stop-query_start).total_seconds()/3600)-1
            distances, indices = index.search(query_embedding, initial_k)
            
            # Filter to remove overlapping results
            filtered_indices = []
            filtered_distances = []
            
            for i, idx in enumerate(indices[0]):
                # Skip the result if it's the query itself
                if abs(distances[0][i]) < 1e-5 and pd_data.index[idx] == query_start:
                    continue
                    
                # Check if this result overlaps with any previously selected result
                overlaps = False
                for prev_idx in filtered_indices:
                    time_diff = abs((pd_data.index[idx] - pd_data.index[prev_idx]).total_seconds() / 3600)
                    if time_diff < min_hours_between_results:
                        overlaps = True
                        break
                        
                if not overlaps:
                    filtered_indices.append(idx)
                    filtered_distances.append(distances[0][i])
                    
                if len(filtered_indices) >= top_k:
                    break
            
            search_succeeded = True
            
            # Prepare results
            for i, (dist, idx) in enumerate(zip(filtered_distances, filtered_indices)):
                start_time = timestamps[idx]
                window_values = pd_data[columns].iloc[idx:idx + window_size].values
                
                result = {
                    'rank': i + 1,
                    'start_time': start_time,
                    'values': window_values,
                    'distance': float(dist)
                }
                
                # Add formatted time if available
                if 'simple_time_string' in pd_data.columns:
                    result['formatted_time'] = pd_data.loc[start_time, 'simple_time_string']
                
                results.append(result)
        else:
            print(f"Not enough data points in the query window. Found {len(query_df)}, need {window_size}")
    else:
        print(f"Query window is too small. It should span at least {window_size} hours")
    
    return results, search_succeeded