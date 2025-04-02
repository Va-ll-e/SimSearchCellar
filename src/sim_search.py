"""Similarity Search Module."""
from datetime import timedelta

import faiss
import numpy as np


def find_similar(pd_data, columns, top_k=5, window_size=3, query_start=None, min_separation=None):
    """Find similar windows of data based on the given query period.

    Args:
        pd_data: DataFrame containing the data with datetime index
        columns: List of column names to include in embeddings
        top_k: Number of similar windows to return (default: 5)
        window_size: Size of the sliding window (default: 3)
        query_start: Start datetime for the query window
        min_separation: Minimum separation between similar windows (default: window_size)

    Returns:
        results: List of dictionaries containing similar windows information
        search_succeeded: Boolean indicating if the search was successful
    """
    # Set default min_separation if not provided
    if min_separation is None:
        min_separation = window_size

    # Input validation
    if pd_data.empty:
        return [], False, "Empty input data"

    if not all(col in pd_data.columns for col in columns):
        missing = [col for col in columns if col not in pd_data.columns]
        return [], False, f"Missing columns: {missing}"

    query_stop = query_start + timedelta(hours=window_size - 1)

    # Validate query window
    if query_start is None or query_stop is None:
        return [], False, "Both query_start and query_stop must be provided or calculable"

    query_df = pd_data.loc[query_start:query_stop]
    if len(query_df) < window_size:
        return [], False, f"Not enough data points in the query window. Found {len(query_df)}, need {window_size}"

    # Extract all relevant data at once instead of window by window
    data_array = pd_data[columns].values
    n_samples = len(pd_data) - window_size + 1

    # Create more efficient window embeddings using strided operations
    # Shape: (n_samples, window_size, n_features)
    windowed_data = np.lib.stride_tricks.sliding_window_view(data_array, (window_size, data_array.shape[1]))[:, 0, :, :]

    # Reshape to (n_samples, window_size * n_features)
    embeddings = windowed_data.reshape(n_samples, -1).astype("float32")

    timestamps = pd_data.index[: -window_size + 1] if n_samples > 0 else []

    # Create FAISS index - for large datasets, consider using approximate search
    if len(embeddings) > 10000:  # Threshold where approximate might be better
        # For large datasets, use IVF index for better performance
        nlist = min(int(np.sqrt(len(embeddings))), 100)  # Rule of thumb
        quantizer = faiss.IndexFlatL2(embeddings.shape[1])
        index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], nlist)
        index.train(embeddings)
        index.add(embeddings)
    else:
        # For smaller datasets, use flat index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

    results = []

    # Create query embedding and search
    query_data = query_df[columns].values
    if len(query_data) > window_size:
        query_data = query_data[:window_size]
    query_data = query_data.flatten().astype("float32")

    # Ensure query has correct shape
    if query_data.shape[0] != embeddings.shape[1]:
        return [], False, f"Query shape mismatch: {query_data.shape[0]} vs expected {embeddings.shape[1]}"

    # Increase the search radius to ensure we have enough candidates after filtering
    distances, indices = index.search(np.array([query_data]), top_k * 5)

    # More efficient filtering using a range-based approach
    filtered_indices = []
    filtered_distances = []

    # Find query index in the dataset
    query_idx = -1
    if query_start is not None:
        try:
            # Find the index corresponding to query_start
            start_loc = pd_data.index.get_loc(query_start)
            if start_loc + window_size <= len(pd_data):
                query_idx = start_loc
        except KeyError:
            # If query_start is not exactly in the index
            pass

    # Create a boolean mask of positions we can't use
    invalid_positions = np.zeros(len(embeddings), dtype=bool)

    # Mark query position as invalid if found
    if query_idx >= 0:
        start_idx = max(0, query_idx - min_separation + 1)
        end_idx = min(len(invalid_positions), query_idx + min_separation)
        invalid_positions[start_idx:end_idx] = True

    for i, idx in enumerate(indices[0]):
        if idx >= len(invalid_positions) or invalid_positions[idx]:
            continue

        filtered_indices.append(idx)
        filtered_distances.append(distances[0][i])

        # Mark nearby positions as invalid
        start_idx = max(0, idx - min_separation + 1)
        end_idx = min(len(invalid_positions), idx + min_separation)
        invalid_positions[start_idx:end_idx] = True

        if len(filtered_indices) >= top_k:
            break

    # Prepare results
    results = []
    # Add basic stats if needed
    for i, (dist, idx) in enumerate(zip(filtered_distances, filtered_indices, strict=False)):
        window = pd_data[columns].iloc[idx : idx + window_size]

        result = {
            "rank": i + 1,
            "start_time": timestamps[idx],
            "end_time": pd_data.index[idx + window_size - 1],
            "distance": round(float(dist), 4),
            "means": {col: round(window[col].mean(), 2) for col in columns},
        }
        results.append(result)

    return results, bool(filtered_indices)
