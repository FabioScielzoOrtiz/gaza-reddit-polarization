import os
import polars as pl

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
processed_data_with_features_path = os.path.join(project_path, 'data', 'processed_data', '04_processed_data.parquet')

base_df = pl.read_parquet(processed_data_with_features_path)

RELEVANCE_THRESHOLD = 3
base_df = base_df.filter(pl.col('content_relevance_score') >= RELEVANCE_THRESHOLD)
print(f'Relevant data records: {len(base_df)}')

final_processed_data_path = os.path.join(project_path, 'data', 'processed_data', '05_processed_data.parquet')
base_df.write_parquet(final_processed_data_path)