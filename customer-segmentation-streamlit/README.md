# Customer Segmentation Analysis - Streamlit Application

A comprehensive web application for customer segmentation analysis using K-means clustering on e-commerce smartphone data.

## Features

### 📊 Multiple Clustering Approaches
- **Visit-Based Clustering**: Segment users based on website visit frequency
- **Price-Based Clustering**: Group users by average purchase price behavior
- **Price-Event Clustering**: Combined analysis of price patterns and event types
- **Brand-Event Clustering**: Clustering based on brand preferences and interaction types
- **Advanced Behavior Clustering**: Comprehensive analysis using visit frequency, conversion ratios, and brand diversity

### 🎯 Interactive Analysis
- Real-time clustering with adjustable parameters
- Elbow method visualization for optimal cluster selection
- Interactive plotly charts and visualizations
- Detailed cluster profiling and business insights
- Comprehensive data overview and statistics

### 💡 Business Intelligence
- Automated cluster profiling with business-relevant names
- Actionable recommendations for each customer segment
- Correlation analysis between user behavior features
- Distribution analysis of key metrics

## Project Structure

```
customer-segmentation-streamlit
├── src
│   ├── app.py                  # Main Streamlit application with all clustering approaches
│   ├── components              # Modular components (optional, functionality now in app.py)
│   │   ├── __init__.py
│   │   ├── data_loader.py      # Functions to load and preprocess customer data
│   │   ├── clustering.py        # Clustering logic and algorithms
│   │   ├── visualizations.py    # Visualization functions for segmentation results
│   │   └── metrics.py          # Metrics for clustering performance
│   ├── utils                   # Utility functions for preprocessing and helpers
│   │   ├── __init__.py
│   │   ├── preprocessing.py     # Data preprocessing utilities
│   │   └── helpers.py          # General helper functions
│   └── config                  # Configuration settings for the application
│       ├── __init__.py
│       └── settings.py         # Application configuration settings
├── data
│   ├── raw                     # Directory for raw data files
│   │   └── .gitkeep
│   └── processed               # Directory for processed data files
│       └── .gitkeep
├── requirements.txt            # Project dependencies
├── streamlit_config.toml       # Streamlit application configuration
└── README.md                   # Project documentation
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 1. Navigate to the Project Directory
```bash
cd customer-segmentation-streamlit
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Data Setup
Ensure your data file `mytestdata.parquet` is in the parent directory:
```
Group-5-E-Commerce-Customer-Segmentation/
├── mytestdata.parquet                    # Your processed data file
├── customer-segmentation-streamlit/
│   ├── src/
│   │   └── app.py                       # Main application
│   ├── requirements.txt
│   └── README.md
```

### 4. Run the Application
```
streamlit run src/app.py
```

This will start the Streamlit server, and you can access the application in your web browser at `http://localhost:8501`.

## Features

- Load and preprocess customer data from various formats (CSV, Parquet).
- Perform K-means clustering on customer visit data.
- Visualize clustering results with histograms, scatter plots, and cluster distributions.
- Calculate and display clustering performance metrics such as silhouette scores and inertia.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.