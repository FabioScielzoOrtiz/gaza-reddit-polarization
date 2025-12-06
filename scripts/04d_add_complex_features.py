
# TODO: pedirle a gemini que lo comente (en ingles, siguiendo el estilo de los otros scripts) y lo mejore (si fuera necesario)

import os, sys
import polars as pl

# --- PATH CONFIGURATION ---
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')

features_dir = os.path.join(project_path, 'data', 'features')
processed_data_dir = os.path.join(project_path, 'data', 'processed_data')
base_data_path = os.path.join(processed_data_dir, '03d_processed_data.parquet')
processed_data_path = os.path.join(processed_data_dir, '04d_processed_data.parquet')



sys.path.append(project_path)

from config.config_03bcd_04bc import (
    FEATURE_CONFIG
)

df_base = pl.read_parquet(base_data_path)

for filename in os.listdir(features_dir):
    feature_name = filename.split('.')[0]
    if feature_name != 'content_relevance_score':
        feature_path = os.path.join(features_dir, filename)
        feature_df = pl.read_parquet(feature_path)
        df_base = df_base.join(feature_df, how='left', on='comment_id')

df_base.write_parquet(processed_data_path)

print('✅ Processing completed successfully.')