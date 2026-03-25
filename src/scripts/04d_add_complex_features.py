
#################################################################################################

# --- IMPORTS ---

import os, sys, logging
import polars as pl

#################################################################################################

# --- LOGGING CONFIGURATION ---

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#################################################################################################

# --- PATH SETUP ---

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..')
sys.path.append(project_path)

features_dir = os.path.join(project_path, 'data', 'features')
processed_data_dir = os.path.join(project_path, 'data', 'processed_data')
base_data_path = os.path.join(processed_data_dir, '03d_processed_data.parquet')
processed_data_path = os.path.join(processed_data_dir, '04d_processed_data.parquet')

#################################################################################################

# --- MAIN EXECUTION ---

def main(): 
       
    df = pl.read_parquet(base_data_path)

    for filename in os.listdir(features_dir):
        feature_name, extension = os.path.splitext(filename) 
        if feature_name != 'content_relevance_score' and extension == '.parquet':
            feature_path = os.path.join(features_dir, filename)
            feature_df = pl.read_parquet(feature_path)
            feature_df = feature_df[['comment_id', feature_name]]
            df = df.join(feature_df, how='left', on='comment_id')

    df.write_parquet(processed_data_path)

    logging.info(f'✅ Complex features added successfully.')
    logging.info(f'📁 Processed file saved at {processed_data_path}.')

#################################################################################################

if __name__ == "__main__":
    main()

#################################################################################################