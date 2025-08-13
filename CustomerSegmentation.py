import pandas as pd
import pyarrow.parquet as pq


# df = pd.read_csv('./2019-Oct.csv')
# df.to_parquet('customer_data.parquet', index=False, engine='pyarrow')

customer_data = pd.read_parquet('customer_data.parquet')
print(customer_data.head())