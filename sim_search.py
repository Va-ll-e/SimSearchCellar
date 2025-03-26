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
    
    # Create window embeddings
    embeddings = []
    timestamps = []
    for i in range(len(pd_data) - window_size + 1):
        window_data = pd_data.iloc[i:i+window_size][columns].values.flatten()
        embeddings.append(window_data)
        timestamps.append(pd_data.index[i])
    embeddings = np.array(embeddings, dtype='float32')
    
    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    results = []
    
    # Validate query window
    if query_stop - query_start < timedelta(hours=window_size-1):
        print(f"Query window is too small. It should span at least {window_size} hours")
        return results, False
        
    query_df = pd_data.loc[query_start:query_stop]
    if len(query_df) < window_size:
        print(f"Not enough data points in the query window. Found {len(query_df)}, need {window_size}")
        return results, False
    
    # Create query embedding and search
    query_data = query_df[columns].iloc[:window_size].values.flatten()
    distances, indices = index.search(np.array([query_data], dtype='float32'), top_k * 3)
    
    # Filter overlapping results
    filtered_indices = []
    filtered_distances = []
    min_hours_between = (abs(query_stop-query_start).total_seconds()/3600)-1
    
    for i, idx in enumerate(indices[0]):
        # Skip if it's the query itself
        if abs(distances[0][i]) < 1e-5 and pd_data.index[idx] == query_start:
            continue
            
        # Check overlap with previous results
        if not any(abs((pd_data.index[idx] - pd_data.index[prev_idx]).total_seconds() / 3600) < min_hours_between 
                  for prev_idx in filtered_indices):
            filtered_indices.append(idx)
            filtered_distances.append(distances[0][i])
            
        if len(filtered_indices) >= top_k:
            break
    
    # Prepare results
    for i, (dist, idx) in enumerate(zip(filtered_distances, filtered_indices)):
        window = pd_data[columns].iloc[idx:idx + window_size]
        result = {
            'rank': i + 1,
            'start_time': timestamps[idx],
            'distance': float(dist),
            **{col: window[col].tolist() for col in columns}
        }
        results.append(result)
    
    return results, bool(filtered_indices)