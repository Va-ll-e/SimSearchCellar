# SimSearchCellar

## Overview
SimSearchCellar is a tool designed to analyze temperature data (temperature sensor sits in a cellar). It uses AI-powered similarity search (FAISS) to identify similar patterns across time series temperature data.

## Problem Statement
This cellar temperature data often shows sudden drops. This tool helps identify these patterns and find similar occurrences in historical data that might indicate system issues or environmental factors.

## How It Works
The system uses a sliding window approach to create embeddings of temperature patterns. These embeddings are then indexed using FAISS for efficient similarity search. When you specify a query window (a period of interest with a temperature drop), the algorithm returns the most similar patterns from other time periods.

## Output
- Visual plots of temperature data with highlighted deviations
- Detailed statistics on temperature management performance
- Similarity search results showing comparable temperature patterns

## Extending the Project
You can modify the similarity search parameters or add additional data sources by updating the code in `sim_search.py` and `plot.py`.