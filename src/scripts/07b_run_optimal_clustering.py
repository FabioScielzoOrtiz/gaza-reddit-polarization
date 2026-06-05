#################################################################################################

# --- IMPORTS ---

import os, sys
import logging
import polars as pl
import numpy as np
import joblib
from kmedoids import KMedoids
from db_robust_clust.models import SampleDistClustering

#################################################################################################

# --- LOGGING CONFIGURATION ---

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#################################################################################################

# --- PATH CONFIGURATION ---

# Uso de os.path.dirname(os.path.abspath(__file__)) para mantener la coherencia con el script 1
script_path = os.path.dirname(os.path.abspath(__file__)) 
project_path = os.path.join(script_path, '..', '..')

# Input / Output:
models_dir = os.path.join(project_path, 'models')
processed_data_dir = os.path.join(project_path, 'data', 'processed_data')
processed_data_path = os.path.join(processed_data_dir, '06_processed_data.parquet')
model_save_path = os.path.join(models_dir, 'optimal_clust_model.joblib')

# Asegurar que el directorio de modelos existe
os.makedirs(models_dir, exist_ok=True)

#################################################################################################

# --- MAIN EXECUTION ---

def main():

    # 1. Load Data
    try:
        processed_data = pl.read_parquet(processed_data_path)
        logging.info(f"Base dataset loaded: {len(processed_data)} records.")
    except Exception as e:
        logging.error(f"❌ Failed to load processed data: {e}")
        exit()

    # 2. Configure Features and Parameters
    try:
        embeddings_cols = [col for col in processed_data.columns if 'embedding' in col]

        numerical_cols = ['sentiment_score'] + embeddings_cols
        ordinal_cols = ['argument_quality_score', 'political_stance_score']
        nominal_cols = ['discourse_tone_score', 'dominant_frame_score']

        QUANT_COLS = numerical_cols
        BINARY_COLS = []
        MULTICLASS_COLS = nominal_cols + ordinal_cols

        N_CLUSTERS = 4
        KMEDOIDS_METHOD = 'pam'
        METRIC = 'ggower'
        D1 = 'robust_mahalanobis'
        D2 = 'sokal'
        D3 = 'hamming'
        ROBUST_METHOD = 'trimmed'
        ALPHA = 0.05
        FRAC_SAMPLE_SIZE = 0.15

        p1 = len(QUANT_COLS)
        p2 = len(BINARY_COLS)
        p3 = len(MULTICLASS_COLS)

        X = processed_data.select(QUANT_COLS + BINARY_COLS + MULTICLASS_COLS)
        
        logging.info(f"✅ Features extracted and configured successfully. Shape of X: {X.shape}.")

    except Exception as e:
        logging.error(f"❌ Error during feature configuration: {e}")
        exit()

    # 3. Train Clustering Model
    try:
        logging.info("Initializing and fitting SampleDistClustering model. This may take a while...")
        
        clustering_method = KMedoids(
            n_clusters=N_CLUSTERS, 
            metric='precomputed', 
            method=KMEDOIDS_METHOD, 
            init='build', 
            max_iter=100, 
            random_state=123
        )

        clust_object = SampleDistClustering(
            clustering_method=clustering_method,
            metric=METRIC,
            frac_sample_size=FRAC_SAMPLE_SIZE,
            random_state=123,
            stratify=False,
            p1=p1, p2=p2, p3=p3,
            d1=D1, d2=D2, d3=D3, 
            robust_method=ROBUST_METHOD, alpha=ALPHA
        )

        clust_object.fit(X)
        logging.info("✅ Model fitted successfully.")

    except Exception as e:
        logging.error(f"❌ Error during model training: {e}")
        exit()

    # 4. Save Model
    try:
        joblib.dump(clust_object, model_save_path)
        logging.info(f'📁 Optimal clustering model saved at {model_save_path}.')
    except Exception as e:
        logging.error(f"❌ Failed to save the model: {e}")
        exit()

if __name__ == "__main__":
    main()

#################################################################################################