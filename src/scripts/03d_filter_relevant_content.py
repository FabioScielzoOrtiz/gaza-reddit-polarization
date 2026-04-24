#################################################################################################

# --- IMPORTS ---

import os, sys, logging
import polars as pl

#################################################################################################

# --- PATH SETUP ---
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..')
sys.path.append(project_path)

features_dir = os.path.join(project_path, 'data', 'features')
processed_data_dir = os.path.join(project_path, 'data', 'processed_data')
feature_file_path = os.path.join(features_dir, 'content_relevance_score.parquet')
base_data_path = os.path.join(processed_data_dir, '02_processed_data.parquet')
processed_data_path = os.path.join(processed_data_dir, '03d_processed_data.parquet')

#################################################################################################

# --- IMPORTS ---

from config.config_03bcd_04bc import (
    FEATURE_CONFIG
)

FEATURE_NAME = 'content_relevance_score'
RELEVANCE_CUTOFF =  FEATURE_CONFIG[FEATURE_NAME]['cutoff']

#################################################################################################

# --- MAIN EXECUTION ---

def main():

    df = pl.read_parquet(base_data_path)
    feature_df = pl.read_parquet(feature_file_path)
    processed_df = df.join(feature_df, how='left', on='comment_id')
    processed_df = processed_df.filter(pl.col(FEATURE_NAME) >= RELEVANCE_CUTOFF)
    processed_df.write_parquet(processed_data_path)
    logging.info(f'⚙️ ✅ Relevant content filtered successfully. Processed file saved at {processed_data_path}.')

if __name__ == "__main__":
    main()