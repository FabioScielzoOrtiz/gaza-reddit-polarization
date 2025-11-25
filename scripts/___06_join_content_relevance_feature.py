import os
import polars as pl

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
processed_data_path = os.path.join(project_path, 'data', 'processed_data', '02_processed_data.parquet')

feature_df = {}
features_dir = os.path.join(project_path, 'data', 'features')
for features_filename in os.listdir(features_dir):
    feature_path = os.path.join(features_dir, features_filename)
    feature_df[features_filename.split('.')[0]] = pl.read_parquet(feature_path)

processed_data = pl.read_parquet(processed_data_path)

for feature_name in feature_df.keys():
    processed_data = processed_data.join(feature_df[feature_name], on='comment_id', how='left')

processed_data_with_features_path = os.path.join(project_path, 'data', 'processed_data', '04_processed_data.parquet')
processed_data.write_parquet(processed_data_with_features_path)