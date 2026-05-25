#################################################################################################

# --- IMPORTS ---

import os, sys, logging, time
import asyncio # New
from openai import AsyncOpenAI # New
from dotenv import load_dotenv

#################################################################################################

# --- LOGGING CONFIGURATION ---

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#################################################################################################

# --- PATH SETUP ---

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..')
sys.path.insert(0, project_path)

labeling_dir = os.path.join(project_path, 'data', 'labeled_samples')
val_sample_path = os.path.join(labeling_dir, '04a_validatION_sample.json')
validation_results_dir = os.path.join(project_path, 'data', 'validation_results')

#################################################################################################

# --- IMPORTS ---

from config.config_04abc import (
    FEATURES_TO_VALIDATE
)

from config.config_03bcd_04bc import (
    FEATURE_CONFIG
)

from config.config_03b_04b import (
    N_VALIDATION_ITERATIONS, 
    GLOBAL_VALIDATION_THRESHOLD,
    MAX_CONCURRENT_REQUESTS,
    BATCH_SIZE
)

from config.config_03bc_04bc_05a import (    
    LLM_MODEL_NAME,
    LLM_TEMPERATURE
)

# Import LLM function (src updated to async)
from src.utils.feature_engineering_utils import (
    load_labeled_sample,
    run_validation_for_feature 
)
#################################################################################################

# --- LOAD ENVIRONMENTAL VARIABLES (OpenAI API) ---

load_dotenv()

#################################################################################################

# --- MAIN EXECUTION ---

async def main():
    
    ###########################################################################

    logging.info("🚀 STARTING COMPLEX FEATURES VALIDATION")
    
    # Init Async Client
    try:
        client = AsyncOpenAI()
    except Exception:
        logging.error("❌ OpenAI Client failed.")
        exit()

    ###########################################################################
    
    # Load Data
    df_val = load_labeled_sample(val_sample_path)

    ###########################################################################

    for feature_name in FEATURES_TO_VALIDATE:

        feature_config = FEATURE_CONFIG.get(feature_name)

        await run_validation_for_feature(
            feature_name, 
            feature_config, 
            df_val, 
            validation_results_dir,
            client, 
            LLM_MODEL_NAME,
            LLM_TEMPERATURE,
            N_VALIDATION_ITERATIONS,
            GLOBAL_VALIDATION_THRESHOLD, 
            MAX_CONCURRENT_REQUESTS,
            BATCH_SIZE
        )

        print('Sleeping...')
        time.sleep(5) # para prevenir saturación entre llamadas

#################################################################################################

if __name__ == "__main__":
    asyncio.run(main())

#################################################################################################