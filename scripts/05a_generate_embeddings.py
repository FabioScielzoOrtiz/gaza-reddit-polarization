#################################################################################################

# --- IMPORTS ---

import os, sys
import logging
import asyncio
import polars as pl
from openai import AsyncOpenAI
from dotenv import load_dotenv

#################################################################################################

# --- LOGGING CONFIGURATION ---

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#################################################################################################

# --- PATH CONFIGURATION ---

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
sys.path.append(project_path)

#################################################################################################

# --- IMPORTS ---

from config.config_05a import (    
    EMBEDDING_MODEL,  
    BATCH_SIZE,
    MAX_CONCURRENT_REQUESTS
)

from src.feature_engineering_utils import run_embedding_generation

#################################################################################################

# --- PATH CONFIGURATION ---

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
sys.path.append(project_path)

# Input: Datos procesados
processed_data_path = os.path.join(project_path, 'data', 'processed_data', '04d_processed_data.parquet')

# Output Final: Embeddings crudos 
raw_embeddings_path = os.path.join(project_path, 'data', 'features', 'raw_embeddings.parquet')

#################################################################################################

# --- LOAD ENVIRONMENTAL VARIABLES ---

load_dotenv()

#################################################################################################

# --- MAIN EXECUTION ---

async def main():
    
    logging.info("🚀 STARTING EMBEDDINGS GENERATION + PCA")

    # 1. CARGAR DATOS
    try:
        df = pl.read_parquet(processed_data_path)
        logging.info(f"📂 Data Loaded: {len(df)} records ready for embedding.")
    except Exception as e:
        logging.error(f"❌ Failed to load data: {e}")
        exit()

    # 2. INICIALIZAR CLIENTE
    try:
        client = AsyncOpenAI()
    except Exception as e:
        logging.error(f"❌ OpenAI Client Error: {e}")
        exit()

    await run_embedding_generation(
        raw_embeddings_path, 
        df, 
        BATCH_SIZE, 
        MAX_CONCURRENT_REQUESTS, 
        EMBEDDING_MODEL, 
        client
    ) 

if __name__ == "__main__":
    asyncio.run(main())

#################################################################################################