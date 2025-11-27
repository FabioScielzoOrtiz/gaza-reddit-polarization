import os, sys
import polars as pl
import logging
from openai import OpenAI
from dotenv import load_dotenv

# --- PATH CONFIGURATION ---
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
sys.path.append(project_path)

# --- CONFIGURATION ---
from config.config_03c_04c import (
    # Set to True to process only a small sample (e.g., 10 records).
    # Set to False to process the entire dataset.
    # Since we use the same output file, you can run Pilot -> Check -> Production (Resume) seamlessly.
    PILOT_MODE, 
    PILOT_SIZE, 
    PILOT_SEED,
    # Save progress every N records 
    BATCH_SAVE_SIZE
)
from config.config_04abc import (
    FEATURES_TO_GENERATE
)
from config.config_03bc_04bc import (
    FEATURE_CONFIG
)


# 1. Input Data (Full processed dataset)
processed_data_path = os.path.join(project_path, 'data', 'processed_data', '03d_processed_data.parquet')

# 2. Training Data (Expert samples for Few-Shot Learning)
train_sample_path = os.path.join(project_path, 'data', 'labeled_samples', '04a_train_sample_relevance.json')

# 3. Output File
features_dir = os.path.join(project_path, 'data', 'features')
os.makedirs(features_dir, exist_ok=True)

# Import Utils
from src.feature_engineering_utils import load_labeled_sample, run_generation_for_feature

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
load_dotenv()


# --- MAIN EXECUTION ---

def main():

    try:
        df = pl.read_parquet(processed_data_path)
        logging.info(f"📂 Base dataset loaded: {len(df)} records.")
    except Exception as e:
        logging.error(f"❌ Failed to load base data: {e}")
        exit()

    try:
        client = OpenAI()
    except Exception as e:
        logging.error(f"❌ OpenAI Client Error: {e}")
        exit()

    df_train = load_labeled_sample(train_sample_path)
     
    for feature_name in FEATURES_TO_GENERATE:

        feature_file_path = os.path.join(features_dir, f'{feature_name}.parquet')

        feature_config = FEATURE_CONFIG.get(feature_name)

        run_generation_for_feature(feature_name, feature_file_path, feature_config, df, df_train, 
                                   BATCH_SAVE_SIZE, PILOT_MODE, PILOT_SIZE, PILOT_SEED, client, logging)

if __name__ == "__main__":
    main()