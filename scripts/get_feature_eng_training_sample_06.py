import os
import json
import polars as pl

SAMPLE_N = 10
SAMPLE_SEED = 123

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
processed_data_path = os.path.join(project_path, 'data', 'processed_data', '05_processed_data.parquet')

processed_data = pl.read_parquet(processed_data_path)

data_sample = processed_data.sample(n=SAMPLE_N, seed=SAMPLE_SEED)

expert_sample = []

data_columns_to_save = ['comment_id', 'post_title', 'post_body', 'comment_body']
features_eng_columns = ['political_stance', 'sentiment_score', 'discourse_tone', 'dominant_frame', 'argument_quality_score']

for row in data_sample[data_columns_to_save].iter_rows():
    row_dict = {col: row[i] for i, col in enumerate(data_columns_to_save)}
    row_dict.update({col: None for col in features_eng_columns})
    expert_sample.append(row_dict)

expert_sample_dir = os.path.join(project_path, 'data', 'expert_sample')
expert_sample_path = os.path.join(expert_sample_dir, 'expert_sample.json')
os.makedirs(expert_sample_dir, exist_ok=True)

try:
    with open(expert_sample_path, 'w', encoding='utf-8') as f:
        json.dump(expert_sample, f, indent=4, ensure_ascii=False)
except Exception as e:
    print(e)