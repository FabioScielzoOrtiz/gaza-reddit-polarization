#################################################################################################

# --- IMPORTS ---

import os
import json
import logging
import polars as pl
import numpy as np

#################################################################################################

# --- LOGGING CONFIGURATION ---

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#################################################################################################

# --- PATH CONFIGURATION ---

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')

# Input paths
features_dir = os.path.join(project_path, 'data', 'features')

# Output paths
generation_cost_dir = os.path.join(project_path, 'data', 'generation_cost')
models_pricing_path = os.path.join(generation_cost_dir, 'models_pricing.csv')
os.makedirs(generation_cost_dir, exist_ok=True)

#################################################################################################

# --- FUNCTIONS ---

def load_feature_metadata(features_dir):
    """
    Carga los metadatos JSON de las features generadas y los convierte a DataFrames.
    """
    logging.info(f"📂 Loading feature metadata from: {features_dir}")
    feature_metadata_dfs = {}
    
    try:
        files = [f for f in os.listdir(features_dir) if f.endswith(".json")]
        if not files:
            logging.warning("⚠️ No JSON metadata files found in features directory.")
            return {}

        for filename in files:
            name, _ = os.path.splitext(filename)
            file_path = os.path.join(features_dir, filename)
            
            with open(file_path, "r", encoding="utf-8") as f:
                feature_metadata_dict = json.load(f)
            
            # Limpiar nombre (quitar sufijo _metadata)
            clean_name = name.replace('_metadata', '')
            feature_metadata_dfs[clean_name] = pl.DataFrame(list(feature_metadata_dict.values()))

        # Ajuste específico para embeddings (output tokens son 0)
        if 'embeddings' in feature_metadata_dfs:
            feature_metadata_dfs['embeddings'] = feature_metadata_dfs['embeddings'].with_columns(
                pl.lit(0).alias('output_tokens')
            )
            logging.info("ℹ️ 'embeddings' metadata adjusted (output_tokens set to 0).")
        
        logging.info(f"✅ Loaded metadata for {len(feature_metadata_dfs)} features.")
        return feature_metadata_dfs

    except Exception as e:
        logging.error(f"❌ Error loading feature metadata: {e}")
        raise e


def calculate_relative_costs(feature_metadata_dfs, models_pricing_df):
    """
    Calcula el coste relativo por muestra para cada feature y modelo.
    """
    logging.info("💰 Calculating relative costs per feature...")
    
    relative_cost_dict = {}
    input_cost, output_cost = {}, {}
    
    # Obtener modelos y sus precios base
    model_names = models_pricing_df['model_name'].unique()
    
    for model_name in model_names:
        model_pricing_df = models_pricing_df.filter(pl.col('model_name') == model_name)
        if not model_pricing_df.is_empty():
            input_cost[model_name] = model_pricing_df['input_cost'][0] 
            output_cost[model_name] = model_pricing_df['output_cost'][0]
        else:
            logging.warning(f"⚠️ Pricing not found for model: {model_name}")

    # Calcular costes por feature
    for model_name in model_names:
        relative_cost_dict[model_name] = {}
        
        # Determinar qué features calcular para este modelo
        if 'embedding' in model_name:
            feature_names = ['embeddings'] if 'embeddings' in feature_metadata_dfs else []
        else:
            feature_names = [x for x in list(feature_metadata_dfs.keys()) if x != 'embeddings']

        for feature_name in feature_names: 
            # Calcular coste total de la feature
            current_df = feature_metadata_dfs[feature_name].with_columns(
                (pl.col('input_tokens') * input_cost[model_name] / 1e6 + 
                 pl.col('output_tokens') * output_cost[model_name] / 1e6).alias('cost')
            ) 
            
            total_cost = current_df['cost'].sum()
            sample_size = current_df.shape[0]
            
            if sample_size > 0:
                relative_cost = total_cost / sample_size
                relative_cost_dict[model_name][feature_name] = round(relative_cost, 6)
            else:
                relative_cost_dict[model_name][feature_name] = 0.0

    # Guardar resultados
    relative_cost_dict_list = [{"model_name": k, **v} for k, v in relative_cost_dict.items()]
    relative_cost_df = pl.DataFrame(relative_cost_dict_list)
    
    relative_cost_file_path = os.path.join(generation_cost_dir, 'relative_cost.csv')
    relative_cost_df.write_csv(relative_cost_file_path)
    
    logging.info(f"💾 Relative costs saved to: {relative_cost_file_path}")
    return relative_cost_df

def total_cost(C1, C2_1, C2_2, n1, n2):
    C = C1 * n1 + (C2_1 + C2_2) * n2 
    return C

def calculate_total_costs(relative_cost_df, n1_list, fil_prop_list):
    """
    Calcula proyecciones de coste total variando el tamaño de la muestra (n1) y filtros (fil_prop).
    """
    logging.info("📊 Calculating total projected costs (Scenario Analysis)...")
    
    feature_names = [col for col in relative_cost_df.columns if col != 'model_name']
    model_names = relative_cost_df['model_name'].unique()
    
    # Separar tipos de features y modelos
    complex_features = [col for col in feature_names if col not in ['embeddings', 'content_relevance_score']]
    llm_model_names = [name for name in model_names if 'embedding' not in name]
    embedding_model_names = [name for name in model_names if name not in llm_model_names]
    
    total_cost_metadata = []
    # Iterar sobre escenarios
    for n1 in n1_list:
        for fil_prop in fil_prop_list:
            n2 = fil_prop * n1
            for llm_model_name in llm_model_names:
                for embedding_model_name in embedding_model_names: 
                    # Coste Fase 1: Relevance Score
                    C1 = relative_cost_df.filter(pl.col('model_name') == llm_model_name).select('content_relevance_score').sum_horizontal()[0]
                    # Coste Fase 2: Features Complejas (LLM) + Embeddings
                    C2_1 = relative_cost_df.filter(pl.col('model_name') == llm_model_name).select(complex_features).sum_horizontal()[0]
                    try:
                        C2_2 = relative_cost_df.filter(pl.col('model_name') == embedding_model_name).select('embeddings').sum_horizontal()[0]
                    except Exception:
                        C2_2 = 0 # Fallback si no hay embeddings
                    # Formula de coste total
                    C = total_cost(C1, C2_1, C2_2, n1, n2)
                    # save total cost metadata
                    total_cost_metadata.append({
                        'llm_model_name': llm_model_name,
                        'embedding_model_name': embedding_model_name,
                        'n1': n1,
                        'fil_prop': fil_prop,
                        'n2': n2, 
                        'total_cost': C
                    })

    total_cost_df = pl.DataFrame(total_cost_metadata)
    total_cost_file_path = os.path.join(generation_cost_dir, 'total_cost.csv')
    total_cost_df.write_csv(total_cost_file_path)
    
    logging.info(f"💾 Total cost projections saved to: {total_cost_file_path}")

#################################################################################################

# --- MAIN EXECUTION ---

def main():

    logging.info("🚀 Starting Cost Calculation Pipeline")

    # 1. Cargar Precios de Modelos
    try:
        if not os.path.exists(models_pricing_path):
            raise FileNotFoundError(f"Pricing file not found at {models_pricing_path}")
            
        models_pricing_df = pl.read_csv(models_pricing_path)
        models_pricing_df = models_pricing_df.fill_null(0)
        logging.info(f"📋 Loaded pricing for {len(models_pricing_df)} models.")
    except Exception as e:
        logging.error(f"❌ Failed to load model pricing: {e}")
        return

    # 2. Cargar Metadata de Features
    try:
        feature_metadata_dfs = load_feature_metadata(features_dir)
        if not feature_metadata_dfs:
             logging.error("❌ No feature metadata loaded. Exiting.")
             return
    except Exception as e:
        logging.error(f"❌ Failed processing feature metadata: {e}")
        return

    # 3. Calcular Costes Relativos
    try:
        relative_cost_df = calculate_relative_costs(feature_metadata_dfs, models_pricing_df)
    except Exception as e:
        logging.error(f"❌ Failed calculating relative costs: {e}")
        return

    # 4. Calcular Costes Totales (Simulación)
    try:
        # Grid the simulation
        n1_list = list(np.arange(start=100000, stop=900000, step=100000)) + [868210]
        fil_prop_list = np.array([0.80, 0.70, 0.60, 0.50])
        calculate_total_costs(relative_cost_df, n1_list, fil_prop_list)
    except Exception as e:
         logging.error(f"❌ Failed calculating total costs: {e}")
         return

    logging.info("✨ Process completed successfully.")

#################################################################################################

if __name__ == "__main__":
    main()

#################################################################################################