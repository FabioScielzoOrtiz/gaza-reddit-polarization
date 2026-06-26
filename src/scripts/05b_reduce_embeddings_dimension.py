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

# Input: Datos procesados
raw_embeddings_path = os.path.join(project_path, 'data', 'features', 'embeddings.parquet')

# Output Final: Features reducidas con PCA
pca_embeddings_path = os.path.join(project_path, 'data', 'features', 'embeddings_pca.parquet')

#################################################################################################

# --- IMPORTS ---

from config.config_05b import (    
    N_PCA_COMPONENTS
)

from src.utils.feature_engineering_utils import run_reduce_embedding_dimension

#################################################################################################

# --- MAIN EXECUTION ---

def main():

    try:
        df_raw_embeddings = pl.read_parquet(raw_embeddings_path)
        logging.info(f"📂 Raw embeddings loaded.")
    except Exception as e:
        logging.error(f"❌ Failed to load data: {e}")
        exit()

    run_reduce_embedding_dimension(
        df_raw_embeddings, 
        pca_embeddings_path
    ) 

if __name__ == "__main__":
    main()

#################################################################################################
