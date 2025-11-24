import os
import json
import polars as pl


script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
processed_data_with_features_path = os.path.join(project_path, 'data', 'processed_data', '04_processed_data.parquet')

processed_data_with_features = pl.read_parquet(processed_data_with_features_path)

SAMPLE_N = 15 if len(processed_data_with_features) > 15 else len(processed_data_with_features)
SAMPLE_SEED = 100
sample = processed_data_with_features.filter(pl.col('content_relevance_score').is_not_null()).sample(n=SAMPLE_N, seed=SAMPLE_SEED)

feature_engineering_validation_sample = []

columns_to_save = ['comment_id', 'post_title', 'post_body', 'comment_body', 'content_relevance_score']

for row in sample[columns_to_save].iter_rows():

    feature_engineering_validation_sample.append({col: row[i] for i, col in enumerate(columns_to_save)})

feature_engineering_validation_sample_dir = os.path.join(project_path, 'data', 'features_validation')
feature_engineering_validation_sample_path = os.path.join(feature_engineering_validation_sample_dir, '03_features_validation.json')
os.makedirs(feature_engineering_validation_sample_dir, exist_ok=True)

try:
    with open(feature_engineering_validation_sample_path, 'w', encoding='utf-8') as f:
        json.dump(feature_engineering_validation_sample, f, indent=4, ensure_ascii=False)
except Exception as e:
    print(e)