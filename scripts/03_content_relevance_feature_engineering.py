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
processed_data_path = os.path.join(project_path, 'data', 'processed_data', '02_processed_data.parquet')
features_dir = os.path.join(project_path, 'data', 'features')
os.makedirs(features_dir, exist_ok=True)
sys.path.append(project_path)

# Add necessary imports for LLM utilities and client initialization here
from src.feature_engineering_utils import (
    content_relevance_score, 
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load the base DataFrame (cleaned data with 'text_content' and 'comment_id').
try:
    df_base = pl.read_parquet(processed_data_path)
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

# Sample is used before positive validation, to save LLM usage cost
N_SAMPLE = 20 if 20 <= len(df_base) else len(df_base)
SAMPLE_SEED = 111
df_base = df_base.sample(n=N_SAMPLE, seed=SAMPLE_SEED)


features_to_generate = list(feature_function_map.keys())
text_content_list = df_base['text_content'].to_list()
comment_id_list = df_base['comment_id'].to_list()


for feature_name in features_to_generate:
   
    output_file_path = os.path.join(features_dir, f'{feature_name}.parquet')
    
    if os.path.exists(output_file_path):
        # 1. Skip if feature already exists (cost optimization).
        logging.info(f"✅ Feature '{feature_name}' exists. Skipping LLM generation.")
        continue

    logging.info(f"⏳ Generating Feature '{feature_name}' with LLM...")
    
    llm_output_list = []
    for text_content in text_content_list:
        llm_output = feature_function_map[feature_name](client=openai_client, content=text_content)
        llm_output_list.append(json.loads(llm_output))

    feature_values = [llm_output[feature_name] for llm_output in llm_output_list]
    
    # 2. Create the feature DataFrame (ID + Score/Category).
    df_feature = pl.DataFrame({
        'comment_id': comment_id_list,
        feature_name: feature_values
    })
    
    # 3. Save the result immediately to ensure persistence.
    df_feature.write_parquet(output_file_path)
    logging.info(f"📁 Feature '{feature_name}' saved at {output_file_path}")

logging.info("--- All feature generation tasks completed. ---")