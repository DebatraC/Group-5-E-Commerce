from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def fill_missing_values(df):
    for column in df.columns:
        if df[column].isnull().any():
            if df[column].dtype == 'object':
                df[column].fillna(df[column].mode()[0], inplace=True)
            else:
                df[column].fillna(df[column].mean(), inplace=True)
    return df

def scale_features(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns)

def preprocess_data(file_path):
    df = pd.read_parquet(file_path)
    df = fill_missing_values(df)
    df = scale_features(df.select_dtypes(include=[np.number]))
    return df