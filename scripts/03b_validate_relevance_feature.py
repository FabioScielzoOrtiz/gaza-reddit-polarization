import os, sys
from openai import OpenAI
from dotenv import load_dotenv

# --- PATH CONFIGURATION ---
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
sys.path.insert(0, project_path)

# --- CONFIGURATION ---
from feature_config.config_03b_04b import (
    FEATURE_CONFIG
)
from feature_config.config_03b import (
    FEATURES_TO_VALIDATE
)

# Directories
labeling_dir = os.path.join(project_path, 'data', 'labeled_samples')
train_sample_path = os.path.join(labeling_dir, '03a_train_sample_relevance.json')
val_sample_path = os.path.join(labeling_dir, '03a_val_sample_relevance.json')
reports_dir = os.path.join(project_path, 'data', 'validation_reports')
os.makedirs(reports_dir, exist_ok=True)

# Import LLM function (Ensure utils is updated)
from src.feature_engineering_utils import (ValidationLogger, 
                                           load_labeled_sample,
                                           run_validation_for_feature 
                                           )

load_dotenv()


# --- MAIN EXECUTION ---

def main():

    # Initialize Logger
    logger = ValidationLogger()

    logger.log("🚀 STARTING MULTI-FEATURE VALIDATION")
    
    # Init Client
    try:
        client = OpenAI()
    except Exception:
        logger.error("❌ OpenAI Client failed.")
        exit()
    
    # Load Data
    df_train = load_labeled_sample(train_sample_path)
    df_val = load_labeled_sample(val_sample_path)

    for feature_name in FEATURES_TO_VALIDATE:

        feature_config = FEATURE_CONFIG.get(feature_name)

        run_validation_for_feature(feature_name, feature_config, df_train, df_val, client, logger)

    # Save final logger
    logger.save()

if __name__ == "__main__":
    main()