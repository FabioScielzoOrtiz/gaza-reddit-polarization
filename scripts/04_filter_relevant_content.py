import os, sys
import polars as pl

# --- PATH CONFIGURATION ---
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')

features_dir = os.path.join(project_path, 'data', 'features')
processed_data_dir = os.path.join(project_path, 'data', 'processed_data')
feature_file_path = os.path.join(features_dir, 'content_relevance_score.parquet')
base_data_path = os.path.join(processed_data_dir, '02_processed_data.parquet')
processed_data_path = os.path.join(processed_data_dir, '04_processed_data.parquet')

sys.path.append(project_path)

from config.config_03b import (
    RELEVANCE_CUTOFF
)

df_base = pl.read_parquet(base_data_path)
feature_df = pl.read_parquet(feature_file_path)
processed_df = df_base.join(feature_df, how='left', on='comment_id')
processed_df = processed_df.filter(pl.col('content_relevance_score') >= RELEVANCE_CUTOFF)
processed_df.write_parquet(processed_data_path)