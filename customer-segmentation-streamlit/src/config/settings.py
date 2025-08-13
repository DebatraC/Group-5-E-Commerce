# settings.py

# Configuration settings for the customer segmentation application

import os

# File paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed/')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'customer_data.parquet')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_customer_data.parquet')

# Clustering parameters
KMEANS_N_CLUSTERS = 4
RANDOM_STATE = 42

# Visualization settings
HISTOGRAM_BINS = 30
SCATTER_ALPHA = 0.6

# Streamlit settings
STREAMLIT_TITLE = "Customer Segmentation Analysis"
STREAMLIT_LAYOUT = "wide"  # Options: 'centered', 'wide', 'experimental'