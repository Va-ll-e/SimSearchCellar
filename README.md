# SimSearchCellar

## Overview
SimSearchCellar is a tool designed to analyze temperature data (temperature sensor sits in a cellar) and detect patterns of sudden temperature drops. It uses AI-powered similarity search (FAISS) to identify similar patterns across time series temperature data.

## Problem Statement
Cellar temperature data often shows sudden drops that can affect stored items like wine or food. This tool helps identify these patterns and find similar occurrences in historical data that might indicate system issues or environmental factors.

## Features
- **Similarity Search**: Find similar temperature patterns using FAISS vector similarity
- **Data Visualization**: Plot actual vs. desired temperature with deviation highlighting
- **Pattern Detection**: Automatically identify drops and spikes in temperature
- **Statistical Analysis**: Calculate deviation metrics and performance statistics

## Installation Requirements

## Project Structure
- `sim_search.py`: Core similarity search algorithm
- `plot.py`: Data visualization and analysis
- `data`: Directory containing temperature datasets
    - `actual_temperature.csv`: Measured temperature readings
    - `wanted_temperature.csv`: Target temperature settings
    - `weather_data.csv`: External weather conditions

## Usage
1. Prepare your temperature data in CSV format
2. Configure the similarity search parameters in `plot.py`
3. Run the analysis:

## How It Works
The system uses a sliding window approach to create embeddings of temperature patterns. These embeddings are then indexed using FAISS for efficient similarity search. When you specify a query window (a period of interest with a temperature drop), the algorithm returns the most similar patterns from other time periods.

## Output
- Visual plots of temperature data with highlighted deviations
- Detailed statistics on temperature management performance
- Similarity search results showing comparable temperature patterns

## Extending the Project
You can modify the similarity search parameters or add additional data sources by updating the code in `sim_search.py` and `plot.py`.
