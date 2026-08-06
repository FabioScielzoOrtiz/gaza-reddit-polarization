#################################################################################################

# --- IMPORTS ---

import os, sys
import argparse
import logging
import polars as pl
import joblib
from sklearn.cluster import KMeans
from kmedoids import KMedoids
from db_robust_clust.models import SampleDistClustering

#################################################################################################

# --- LOGGING CONFIGURATION ---

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#################################################################################################

# --- ARGUMENT PARSING ---

def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena los modelos de clustering definidos en clust_config_metadata."
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help="Fuerza la re-ejecución y sobrescritura de los modelos que ya existan en disco."
    )
    return parser.parse_args()

#################################################################################################

# --- PATH CONFIGURATION ---

# Uso de os.path.dirname(os.path.abspath(__file__)) para mantener la coherencia con el script 1
script_path = os.path.dirname(os.path.abspath(__file__)) 
project_path = os.path.join(script_path, '..', '..')

# Input / Output:
models_dir = os.path.join(project_path, 'models')
processed_data_dir = os.path.join(project_path, 'data', 'processed_data')
processed_data_path = os.path.join(processed_data_dir, '06_processed_data.parquet')

# Asegurar que el directorio de modelos existe
os.makedirs(models_dir, exist_ok=True)

sys.path.append(project_path)
from config.config_07b import clust_config_metadata

#################################################################################################

# --- MAIN EXECUTION ---

def main():

    args = parse_args()

    # 1. Load Data
    try:
        processed_data = pl.read_parquet(processed_data_path)
        logging.info(f"Base dataset loaded: {len(processed_data)} records.")
    except Exception as e:
        logging.error(f"❌ Failed to load processed data: {e}")
        exit()
    
    for config_key, config in clust_config_metadata.items():

        logging.info(f'▶️ Running for {config_key}')

        # 1bis. Comprobar si el modelo ya existe para saltar la iteración (salvo --force)
        model_save_path = os.path.join(models_dir, f'optimal_model_{config_key}.joblib')

        if os.path.exists(model_save_path) and not args.force:
            logging.info(
                f"⛔ El modelo para '{config_key}' ya existe en {model_save_path} "
                f"--> Saltando esta iteración (usa --force/-f para forzar la re-ejecución)."
            )
            continue

        QUANT_COLS = config['quant_cols']
        BINARY_COLS = config['binary_cols']
        MULTICLASS_COLS = config['multiclass_cols']
        N_CLUSTERS = config['n_clusters']
        RANDOM_STATE = config['random_state']

        if config_key not in ['clust_config_I_b', 'clust_config_III', 'clust_config_III_b']:
            KMEDOIDS_METHOD = config['kmedoids_method']
            FRAC_SAMPLE_SIZE = config['frac_sample_size']
            METRIC = config['metric']
            D1 = config['d1']
            D2 = config['d2']
            D3 = config['d3']
            ROBUST_METHOD = config['robust_method']
            ALPHA = config['alpha']
            p1 = len(QUANT_COLS)
            p2 = len(BINARY_COLS)
            p3 = len(MULTICLASS_COLS)

        # 2. Configure Features and Parameters
        try:
            
            X = processed_data.select(QUANT_COLS + BINARY_COLS + MULTICLASS_COLS)
            logging.info(f"Features extracted and configured successfully. Shape of X: {X.shape}.")

        except Exception as e:
            logging.error(f"❌ Error during feature configuration: {e}")
            exit()

        # 3. Train Clustering Model
        try:
            logging.info("Initializing and fitting SampleDistClustering model. This may take a while...")
            
            if config_key not in ['clust_config_I_b', 'clust_config_III', 'clust_config_III_b']:

                clustering_method = KMedoids(
                    n_clusters=N_CLUSTERS, 
                    metric='precomputed', 
                    method=KMEDOIDS_METHOD, 
                    init='build', 
                    max_iter=100, 
                    random_state=RANDOM_STATE
                )

                clust_object = SampleDistClustering(
                    clustering_method=clustering_method,
                    metric=METRIC,
                    frac_sample_size=FRAC_SAMPLE_SIZE,
                    random_state=RANDOM_STATE,
                    stratify=False,
                    p1=p1, p2=p2, p3=p3,
                    d1=D1, d2=D2, d3=D3, 
                    robust_method=ROBUST_METHOD, alpha=ALPHA
                )
            
            else:
                clust_object = KMeans(
                n_clusters=N_CLUSTERS,
                random_state=RANDOM_STATE
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