#################################################################################################

# --- IMPORTS ---

import os
import sys
import json
import logging
import polars as pl

#################################################################################################

# --- LOGGING CONFIGURATION ---

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#################################################################################################

# --- PATH CONFIGURATION ---

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
sys.path.insert(0, project_path)
labeling_dir = os.path.join(project_path, 'data', 'labeled_samples')
validation_results_dir = os.path.join(project_path, 'data', 'validation_results')

#################################################################################################

# --- IMPORTS ---

from src.feature_engineering_utils import load_labeled_sample

from config.config_03bcd_04bc import (
    FEATURE_CONFIG
)

#################################################################################################

# --- FUNCTIONS ---

def get_data_for_features_validation_analysis(df_val, val_results, feature_name, feature_type, validation_results_dir):

    true_values_df = df_val[['comment_id', feature_name]]

    iter_pred_values_dict = val_results['llm_metadata']['iterations_predicted_values']
    iter_pred_values_df_wide = pl.DataFrame(iter_pred_values_dict)
    comment_ids = list(iter_pred_values_dict.keys())
    predicted_values = list(iter_pred_values_dict.values())
    
    # DataFrame largo con lista de predicciones por comment_id
    iter_pred_values_df_long = pl.DataFrame({
        'comment_id': comment_ids, 
        'predicted_values': predicted_values
    })
    
    # --- Lógica Diferenciada por Tipo de Feature ---
    
    if feature_type in ['ordinal', 'continuous']:
        # Lógica original para numéricas
        true_predicted_values_df = true_values_df.join(
            iter_pred_values_df_long, on='comment_id', how='inner'
        ).with_columns(
            predicted_values_mean = pl.col('predicted_values').list.mean().round(2),
            predicted_values_std = pl.col('predicted_values').list.std().round(2),
            predicted_values_q25 = pl.col('predicted_values').list.eval(pl.element().quantile(0.25)).list.first().round(2),
            predicted_values_q75 = pl.col('predicted_values').list.eval(pl.element().quantile(0.75)).list.first().round(2),
            predicted_values_range = 5 - 0 
        ).with_columns(
            predicted_values_cv_std = (pl.col('predicted_values_std') / pl.col('predicted_values_mean').abs()).round(2).fill_nan(None),
            predicted_values_cv_quantiles = ((pl.col('predicted_values_q75') - pl.col('predicted_values_q25')) / (pl.col('predicted_values_q75') + pl.col('predicted_values_q25')).abs()).round(2).fill_nan(None),
            predicted_values_cv_range = (pl.col('predicted_values_std') / pl.col('predicted_values_range')).round(2).fill_nan(None)
        ).sort(by='predicted_values_mean')

        # Columnas métricas para el plot
        metric_cols = ['predicted_values_mean', 'predicted_values_std']

    elif feature_type in ['categorical']:
        # LOGICA CORREGIDA PARA CATEGÓRICAS
        
        true_predicted_values_df = true_values_df.join(
            iter_pred_values_df_long, on='comment_id', how='inner'
        ).with_columns(
            # 1. Calculamos la MODA usando list.eval y value_counts
            # value_counts(sort=True) pone el más frecuente primero.
            # Tomamos el primer struct y extraemos el valor (el nombre de campo es "" o el nombre de la col, pero struct[0] es el valor)
            predicted_values_mode = pl.col('predicted_values').list.eval(
                pl.element().value_counts(sort=True).struct[0].first()
            ).list.first(),

            # 2. Calculamos la CONSISTENCIA (Frecuencia del valor más repetido / Total)
            # Extraemos el campo "count" (struct[1]) del primer elemento (el más frecuente)
            predicted_values_consistency = pl.col('predicted_values').list.eval(
                pl.element().value_counts(sort=True).struct.field("count").first() / pl.element().len()
            ).list.first()
        ).with_columns(

            predicted_values_variability = 1 - pl.col('predicted_values_consistency')
            
        ).sort(by='predicted_values_consistency', descending=True)

        metric_cols = ['predicted_values_mode', 'predicted_values_consistency', 'predicted_values_variability']
    
    else:
        raise ValueError(f"Tipo de feature no soportado: {feature_type}")

    # --- Generación del DataFrame para Plot ---
    
    # Hacemos unpivot (melt) para tener una fila por cada predicción individual
    df_plot = iter_pred_values_df_wide.unpivot(
        variable_name="comment_id", 
        value_name="prediction"
    ).sort(
        by='prediction'
    ).join(
        true_values_df, on='comment_id', how='inner'
    ).join(
        # Aquí usamos metric_cols dinámicamente según el tipo de feature
        true_predicted_values_df[['comment_id'] + metric_cols], 
        on='comment_id', 
        how='inner'
    )
    
    # --- Guardado de Archivos ---
    
    # Ajuste: Cambié la extensión a .parquet en la ruta porque usas write_parquet abajo
    true_predicted_values_df_path = os.path.join(validation_results_dir, f'true_predicted_values_{feature_name}.parquet')
    df_plot_path = os.path.join(validation_results_dir, f'plot_true_predicted_values_{feature_name}.csv')
        
    # Guardamos
    true_predicted_values_df.write_parquet(true_predicted_values_df_path)
    df_plot.write_csv(df_plot_path)

    return true_predicted_values_df, df_plot

#################################################################################################

# --- MAIN EXECUTION ---

def main():

    logging.info("🚀 Starting Feature Validation Analysis Pipeline")

    for feature_name in FEATURE_CONFIG.keys(): 

        feature_type = FEATURE_CONFIG[feature_name]['type']

        try:
            filename = '03a_validation_sample.json' if feature_name == 'content_relevance_score' else '04a_validation_sample.json'
            val_sample_path = os.path.join(labeling_dir, filename) 
            if not os.path.exists(val_sample_path):
                raise FileNotFoundError(f"Validation sample file not found at {val_sample_path}")
                
            df_val = load_labeled_sample(val_sample_path)
            logging.info(f"📋 Loaded validation sample.")

        except Exception as e:
            logging.error(f"❌ Failed to load validation sample: {e}")
            return

        try:
            val_results_path = os.path.join(validation_results_dir, f'validation_results_{feature_name}.json')
            if not os.path.exists(val_results_path):
                raise FileNotFoundError(f"Validation results file not found at {val_results_path}")
            
            with open(val_results_path, "r", encoding="utf-8") as f:
                val_results = json.load(f)
            logging.info(f"📋 Loaded validation results.")

        except Exception as e:
            logging.error(f"❌ Failed to load validation results: {e}")
            return

        
        logging.info(f"📥 Generating Data for {feature_name.upper()} Validation Analysis")
        #try: 
        get_data_for_features_validation_analysis(df_val, val_results, feature_name, feature_type, validation_results_dir)
        logging.info("✅ Data generated successfully")
        #except Exception as e:
        #    logging.info(f"❌ Error generating the data: {e}")

#################################################################################################

if __name__ == "__main__":
    main()

#################################################################################################