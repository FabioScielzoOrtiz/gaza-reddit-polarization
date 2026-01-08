#################################################################################################

# --- IMPORTS ---

import os, sys
import json
import polars as pl
import logging
import asyncio # New
from openai import AsyncOpenAI # New
from dotenv import load_dotenv

#################################################################################################

# --- LOGGING CONFIGURATION ---

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#################################################################################################

# --- PATH CONFIGURATION ---

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
sys.path.append(project_path)

processed_data_path = os.path.join(project_path, 'data', 'processed_data', '03d_processed_data.parquet')
train_sample_path = os.path.join(project_path, 'data', 'labeled_samples', '04a_train_sample.json')
validation_results_dir = os.path.join(project_path, 'data', 'validation_results')
features_dir = os.path.join(project_path, 'data', 'features')
os.makedirs(features_dir, exist_ok=True)

#################################################################################################

# --- IMPORTS ---

from config.config_03bc_04bc_05a import (    
    PILOT_MODE, 
    PILOT_SIZE, 
    PILOT_SEED,
    BATCH_SAVE_SIZE,
    MAX_CONCURRENT_REQUESTS,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE
)
from config.config_04abc import (
    FEATURES_TO_GENERATE
)
from config.config_03bcd_04bc import (
    FEATURE_CONFIG
)

# Import Utils
from src.feature_engineering_utils import load_labeled_sample, run_generation_for_feature

#################################################################################################

# --- LOAD ENVIRONMENTAL VARIABLES ---

load_dotenv()

#################################################################################################

# --- MAIN EXECUTION ---

async def main():

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
        client = AsyncOpenAI()
        logging.info("🤖 Async Client Ready")
    except Exception as e:
        logging.error(f"❌ OpenAI Client Error: {e}")
        exit()

    ###########################################################################  
     
    for feature_name in FEATURES_TO_GENERATE:
        
        validation_results_path = os.path.join(validation_results_dir, f"validation_results_{feature_name}.json")
        
        if not os.path.exists(validation_results_path):
            logging.warning(f"⚠️ Validation file not found for {feature_name}. Skipping.")
            continue

        with open(validation_results_path, "r", encoding="utf-8") as f:
            validation_results = json.load(f)

        if validation_results['global_validation'].get('validation_passed', False):

            feature_file_path = os.path.join(features_dir, f'{feature_name}.parquet')
            metadata_file_path = os.path.join(features_dir, f'{feature_name}_metadata.json')
            feature_config = FEATURE_CONFIG.get(feature_name)
            file_lock = asyncio.Lock()

            # Llamada Asíncrona Masiva
            await run_generation_for_feature(
                feature_name, 
                feature_file_path, 
                feature_config, 
                df, 
                df_train, 
                BATCH_SAVE_SIZE, 
                MAX_CONCURRENT_REQUESTS,
                client,
                LLM_MODEL_NAME,
                LLM_TEMPERATURE,
                metadata_file_path, 
                file_lock,
                PILOT_MODE, 
                PILOT_SIZE, 
                PILOT_SEED,
            )
        
        else:
            logging.warning(f'🛑 Validation not passed for {feature_name}. Improve LLM configuration (model, prompt, temperature, few-shot-learning, etc.).')
            logging.warning('⏭️  Generation skipped.')

#################################################################################################

if __name__ == "__main__":
    asyncio.run(main())

#################################################################################################