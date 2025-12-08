#################################################################################################

# --- IMPORTS ---

import os, sys
import json
import polars as pl
import logging
from openai import OpenAI
from dotenv import load_dotenv

#################################################################################################

# --- LOGGING CONFIGURATION ---

# Set up basic configuration to log INFO level messages.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#################################################################################################

# --- PATH CONFIGURATION ---

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
sys.path.append(project_path)

# Input Data path (Full processed dataset)
processed_data_path = os.path.join(project_path, 'data', 'processed_data', '02_processed_data.parquet')

# Training Data path (Expert samples for Few-Shot Learning)
train_sample_path = os.path.join(project_path, 'data', 'labeled_samples', '03a_train_sample_relevance.json')

# Validation results path
validation_results_dir = os.path.join(project_path, 'data', 'validation_results')

# Output File path
features_dir = os.path.join(project_path, 'data', 'features')
os.makedirs(features_dir, exist_ok=True)

#################################################################################################

# --- IMPORTS ---

from config.config_03c_04c import (    
    PILOT_MODE, 
    PILOT_SIZE, 
    PILOT_SEED,
    BATCH_SAVE_SIZE
)
from config.config_03abc import (
    FEATURES_TO_GENERATE
)
from config.config_03bcd_04bc import (
    FEATURE_CONFIG
)

# Import Utils
from src.feature_engineering_utils import load_labeled_sample, run_generation_for_feature

#################################################################################################

# --- LOAD ENVIRONMENTAL VARIABLES (OpenAI API) ---

load_dotenv()

#################################################################################################

# --- MAIN EXECUTION ---

def main():

    ###########################################################################

    try:
        df = pl.read_parquet(processed_data_path)
        logging.info(f"📂 Base dataset loaded: {len(df)} records.")
    except Exception as e:
        logging.error(f"❌ Failed to load base data: {e}")
        exit()

    df_train = load_labeled_sample(train_sample_path)

    ###########################################################################

    try:
        client = OpenAI()
    except Exception as e:
        logging.error(f"❌ OpenAI Client Error: {e}")
        exit()

    ###########################################################################  
     
    for feature_name in FEATURES_TO_GENERATE:
        
        validation_results_path = os.path.join(validation_results_dir, f"validation_results_{feature_name}.json")
        with open(validation_results_path, "r", encoding="utf-8") as f:
            validation_results = json.load(f)

        if validation_results['validation_passed']:

            feature_file_path = os.path.join(features_dir, f'{feature_name}.parquet')

            feature_config = FEATURE_CONFIG.get(feature_name)

            run_generation_for_feature(
                feature_name, 
                feature_file_path, 
                feature_config, 
                df, 
                df_train, 
                BATCH_SAVE_SIZE, 
                PILOT_MODE, 
                PILOT_SIZE, 
                PILOT_SEED, 
                client 
            )
        
        else:

            logging.warning(f'🛑 Validation not passed for {feature_name}.\nPossible Solutions: a) Improve LLM Prompt b) Increase Few-Shot Samples')
            logging.warning(f'⏭️ Generation skipped for {feature_name}')

#################################################################################################

if __name__ == "__main__":
    main()

#################################################################################################