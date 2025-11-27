import os, sys
import logging 
import polars as pl

# --- PATH CONFIGURATION ---
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
sys.path.insert(0, project_path)

from config.config_03a_04a import (
    SAMPLE_N, # Total target number of samples (Manual + Random) 
    SAMPLE_SEED, # Seed for reproducibility
    VAL_SAMPLE_RATIO, # 80% for Blind Validation, 20% for Few-Shot Training
    # Add specific comment_ids here to FORCE them into the specific set. Useful for including known edge cases (sarcasm, short text) in the prompt.
    MANUAL_TRAIN_IDS, 
    MANUAL_VAL_IDS,
    # TODO: add descriptive comment
    DATA_COLUMNS_TO_INCLUDE
)
from config.config_04abc import (
    FEATURES_TO_LABEL
)

from src.feature_engineering_utils import run_labeling_samples


# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Input: Base processed data (from Step 03d)
base_data_path = os.path.join(project_path, 'data', 'processed_data', '03d_processed_data.parquet')

# Output: New dedicated folder for manual labeling inputs
labeling_dir = os.path.join(project_path, 'data', 'labeled_samples')
os.makedirs(labeling_dir, exist_ok=True)

def main():

    train_sample_path = os.path.join(labeling_dir, '04a_train_sample_relevance.json')
    val_sample_path = os.path.join(labeling_dir, '04a_val_sample_relevance.json')

    if os.path.exists(train_sample_path) and os.path.exists(val_sample_path):
        logging.info("⛔ Train and validation samples already exist --> Process Stopped")
        exit()

    try:
        df = pl.read_parquet(base_data_path)
        logging.info(f"📂 Base dataset loaded: {len(df)} records.")
    except Exception as e:
        logging.error(f"❌ Error loading data: {e}")
        exit()

    run_labeling_samples(df, DATA_COLUMNS_TO_INCLUDE, FEATURES_TO_LABEL, 
                         SAMPLE_N, SAMPLE_SEED, VAL_SAMPLE_RATIO, 
                         MANUAL_TRAIN_IDS, MANUAL_VAL_IDS,
                         train_sample_path, val_sample_path)
    
if __name__ == "__main__":
    main()