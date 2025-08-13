import pandas as pd
import dask.dataframe as dd

def load_data(file_path: str, file_type: str = 'parquet') -> pd.DataFrame:
    if file_type == 'parquet':
        data = pd.read_parquet(file_path)
    elif file_type == 'csv':
        data = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file type. Please use 'parquet' or 'csv'.")
    
    return data

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    # Example preprocessing steps
    data.dropna(inplace=True)  # Drop missing values
    # Add more preprocessing steps as needed
    return data

def load_and_preprocess_data(file_path: str, file_type: str = 'parquet') -> pd.DataFrame:
    data = load_data(file_path, file_type)
    processed_data = preprocess_data(data)
    return processed_data