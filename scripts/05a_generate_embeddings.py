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

from config.config_03bc_04bc_05a import (    
    EMBEDDING_MODEL_NAME,  
    BATCH_SIZE,
    MAX_CONCURRENT_REQUESTS,
    PILOT_MODE, 
    PILOT_SIZE, 
    PILOT_SEED,
)

from utils.feature_engineering_utils import run_embedding_generation

#################################################################################################

# --- PATH CONFIGURATION ---

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
sys.path.append(project_path)

features_dir = os.path.join(project_path, 'data', 'features')
metadata_file_path = os.path.join(features_dir, f'embeddings_metadata.json')

# Input: Datos procesados
processed_data_path = os.path.join(project_path, 'data', 'processed_data', '04d_processed_data.parquet')

# Output Final: Embeddings crudos 
raw_embeddings_path = os.path.join(features_dir, 'embeddings.parquet')

#################################################################################################

# --- LOAD ENVIRONMENTAL VARIABLES ---

load_dotenv()

#################################################################################################

# --- MAIN EXECUTION ---

async def main():
    
    logging.info("🚀 STARTING EMBEDDINGS GENERATION + PCA")

    try:
        df = pl.read_parquet(processed_data_path)
        logging.info(f"📂 Data Loaded: {len(df)} records ready for embedding.")
    except Exception as e:
        logging.error(f"❌ Failed to load data: {e}")
        exit()

    try:
        client = AsyncOpenAI()
    except Exception as e:
        logging.error(f"❌ OpenAI Client Error: {e}")
        exit()

    file_lock = asyncio.Lock()

    await run_embedding_generation(
        raw_embeddings_path, 
        df, 
        BATCH_SIZE, 
        MAX_CONCURRENT_REQUESTS, 
        client,
        EMBEDDING_MODEL_NAME, 
        metadata_file_path, 
        file_lock,
        PILOT_MODE, 
        PILOT_SIZE, 
        PILOT_SEED,
    ) 

if __name__ == "__main__":
    asyncio.run(main())

#################################################################################################