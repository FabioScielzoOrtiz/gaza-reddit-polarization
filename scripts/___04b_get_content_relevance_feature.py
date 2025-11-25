import os, sys
import json
import polars as pl
import logging
from openai import OpenAI
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# --- PATH CONFIGURATION ---
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
features_dir = os.path.join(project_path, 'data', 'features')
val_scores_path = os.path.join(features_dir, 'val_scores_04a.json')
base_data_path = os.path.join(project_path, 'data', 'processed_data', '02_processed_data.parquet')
sys.path.append(project_path)

# Add necessary imports for LLM utilities and client initialization here
from src.feature_engineering_utils import (
    content_relevance_score, 
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

is_positive_validation = open()  # defined in 04b

# Load the base DataFrame (cleaned data with 'text_content' and 'comment_id').
try:
    df_base = pl.read_parquet(base_data_path)
    logging.info(f"Loaded base DataFrame with {len(df_base)} records.")
except Exception as e:
    logging.error(f"Failed to load processed data: {e}")
    exit()

# Initialize OpenAI client once
try:
    openai_client = OpenAI()
    openai_client.models.list()
    logging.info("OpenAI client initialized successfully.")
except Exception as e:
    print(e)
    logging.error(f"Failed to initialize OpenAI client. Check your API key. Error: {e}")

# Map feature names to their corresponding generation function.
feature_function_map = {
    'content_relevance_score': content_relevance_score
}

# --- FEATURE LIST AND EXECUTION ---

features_to_generate = list(feature_function_map.keys())

if PILOT_SAMPLE: # For testing purposes, not production
    df_base = df_base.sample(n=N_PILOT_SAMPLE, seed=PILOT_SAMPLE_SEED)

train_df = process_json_to_df(train_sample_path)
train_text_content_list = train_df['text_content'].to_list()
train_comment_id_list = train_df['comment_id'].to_list()

val_df = process_json_to_df(train_sample_path)
val_comment_id_list = val_df['comment_id'].to_list()

# Filter train and val data, since are already predicted
df_base_filtered = df_base.filter(~ pl.col('comment_id').is_in(train_comment_id_list + val_comment_id_list))
predict_text_content_list = df_base_filtered['text_content'].to_list()
predict_comment_id_list = df_base_filtered['comment_id'].to_list()

for feature_name in features_to_generate:

    if is_positive_validation[feature_name]:

        output_file_path = os.path.join(features_dir, f'{feature_name}.parquet')
        
        if os.path.exists(output_file_path):
            # 1. Skip if feature already exists (cost optimization).
            logging.info(f"✅ Feature '{feature_name}' exists. Skipping LLM generation.")
            continue

        logging.info(f"⏳ Generating Feature '{feature_name}' with LLM...")
        
        llm_output_list = []
        for predict_text_content in predict_text_content_list:
            llm_output = feature_function_map[feature_name](client=openai_client, train_data=train_text_content_list, predict_data=predict_text_content)
            llm_output_list.append(json.loads(llm_output))

        predicted_feature_values = [llm_output[feature_name] for llm_output in llm_output_list]
        
        # 2. Create the feature DataFrame (ID + Score/Category).
        train_feature_values = train_df[feature_name].to_list()
        val_feature_values = val_df[feature_name].to_list()
        df_feature = pl.DataFrame({
            'comment_id': predict_comment_id_list + train_comment_id_list + val_comment_id_list,
            feature_name: predicted_feature_values + train_feature_values + val_feature_values
        })
        
        # 3. Save the result immediately to ensure persistence.
        df_feature.write_parquet(output_file_path)
        logging.info(f"📁 Feature '{feature_name}' saved at {output_file_path}")

    else:

        print(f'NEGATIVE VALIDATION for {feature_name}. IMPROVE {feature_name} GENERATION BEFORE CONTINUING.')