import os, sys, logging
from openai import OpenAI
from dotenv import load_dotenv

# --- PATH CONFIGURATION ---
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
sys.path.insert(0, project_path)

# --- CONFIGURATION ---
from config.config_03bcd_04bc import (
    FEATURE_CONFIG
)
from config.config_04abc import (
    FEATURES_TO_VALIDATE
)

# Directories
labeling_dir = os.path.join(project_path, 'data', 'labeled_samples')
train_sample_path = os.path.join(labeling_dir, '04a_train_sample_relevance.json')
val_sample_path = os.path.join(labeling_dir, '04a_val_sample_relevance.json')
validation_results_dir = os.path.join(project_path, 'data', 'validation_results')

# Import LLM function (Ensure utils is updated)
from src.feature_engineering_utils import (
    load_labeled_sample,
    run_validation_for_feature 
)

load_dotenv()


# --- MAIN EXECUTION ---

def main():

    # Initialize logging
    logging.info("🚀 STARTING MULTI-FEATURE VALIDATION")
    
    # Init Client
    try:
        client = OpenAI()
    except Exception:
        logging.error("❌ OpenAI Client failed.")
        exit()
    
    # Load Data
    df_train = load_labeled_sample(train_sample_path)
    df_val = load_labeled_sample(val_sample_path)

    for feature_name in FEATURES_TO_VALIDATE:

        feature_config = FEATURE_CONFIG.get(feature_name)

        run_validation_for_feature(
            feature_name, 
            feature_config, 
            df_train, 
            df_val, 
            client, 
            validation_results_dir
        )

if __name__ == "__main__":
    main()