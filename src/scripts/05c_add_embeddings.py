#################################################################################################

# --- IMPORTS ---

import os, sys
import logging
import polars as pl

#################################################################################################

# --- LOGGING CONFIGURATION ---

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#################################################################################################

# --- PATH CONFIGURATION ---

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..')
sys.path.append(project_path)

# Input:
base_data_path = os.path.join(project_path, 'data', 'processed_data', '04d_processed_data.parquet')
pca_embeddings_path = os.path.join(project_path, 'data', 'features', 'embeddings_pca.parquet')
processed_data_path = os.path.join(project_path, 'data', 'processed_data', '05c_processed_data.parquet')

#################################################################################################

# --- MAIN EXECUTION ---

def main():

    try:
        df = pl.read_parquet(base_data_path)
        logging.info(f"Base dataset loaded: {len(df)} records.")
    except Exception as e:
        logging.error(f"❌ Failed to load base data: {e}")
        exit()

    try:
        df_pca_embeddings = pl.read_parquet(pca_embeddings_path)
        logging.info(f"PCA embeddings dataset loaded: {len(df)} records.")
    except Exception as e:
        logging.error(f"❌ Failed to load PCA embeddings data: {e}")
        exit()

    df = df.join(df_pca_embeddings, how='left', on='comment_id')

    df.write_parquet(processed_data_path)

    logging.info(f'✅ PCA embeddings added successfully.')
    logging.info(f'📁 Processed file saved at {processed_data_path}.')

if __name__ == "__main__":
    main()

#################################################################################################
