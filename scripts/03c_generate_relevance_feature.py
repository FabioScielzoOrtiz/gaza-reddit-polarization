import os, sys
import json
import time
import polars as pl
import logging
from openai import OpenAI
from dotenv import load_dotenv

# --- PATH CONFIGURATION ---
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
sys.path.append(project_path)

# --- CONFIGURATION ---
from config.config_03c import (
    # Set to True to process only a small sample (e.g., 10 records).
    # Set to False to process the entire dataset.
    # Since we use the same output file, you can run Pilot -> Check -> Production (Resume) seamlessly.
    PILOT_MODE, 
    PILOT_SIZE, 
    PILOT_SEED,
    # Save progress every N records 
    BATCH_SAVE_SIZE
)

# 1. Input Data (Full processed dataset)
processed_data_path = os.path.join(project_path, 'data', 'processed_data', '02_processed_data.parquet')

# 2. Training Data (Expert samples for Few-Shot Learning)
train_sample_path = os.path.join(project_path, 'data', 'labeled_samples', '03a_train_sample_relevance.json')

# 3. Output File
features_dir = os.path.join(project_path, 'data', 'features')
os.makedirs(features_dir, exist_ok=True)
feature_file_path = os.path.join(features_dir, 'content_relevance_score.parquet')

# Import Utils
from src.feature_engineering_utils import content_relevance_score

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
load_dotenv()

# --- HELPER FUNCTIONS ---

def process_train_data_for_llm(filepath):
    """Loads expert labeled examples to inject into the prompt."""
    if not os.path.exists(filepath):
        logging.warning(f"⚠️ No labeled training samples found at {filepath}. Running zero-shot.")
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Format for utils function
        formatted_examples = []
        for item in data:
            if item.get('content_relevance_score') is not None:
                formatted_examples.append({
                    "text_content": item.get('text_content', ''),
                    "content_relevance_score": item.get('content_relevance_score')
                })
        logging.info(f"✅ Loaded {len(formatted_examples)} expert examples for Few-Shot Prompting.")
        return formatted_examples
    except Exception as e:
        logging.error(f"❌ Error loading training samples: {e}")
        return None

# --- MAIN EXECUTION ---

def main():
    mode_msg = f"🧪 PILOT MODE (Max {PILOT_SIZE} records)" if PILOT_MODE else "🚀 PRODUCTION MODE (Full Data)"
    logging.info(f"STARTING GENERATION: Content Relevance Feature")
    logging.info(f"MODE: {mode_msg}")

    # 1. Load Base Data
    try:
        df_base = pl.read_parquet(processed_data_path)
        logging.info(f"📂 Base dataset loaded: {len(df_base)} records.")
    except Exception as e:
        logging.error(f"❌ Failed to load base data: {e}")
        exit()

    # 2. Load Context (Few-Shot Examples)
    few_shot_examples = process_train_data_for_llm(train_sample_path)

    # 3. Initialize OpenAI
    try:
        client = OpenAI()
    except Exception as e:
        logging.error(f"❌ OpenAI Client Error: {e}")
        exit()

    # 4. PREPARE DATA (Resume Logic)
    
    # A. Check what is already done
    processed_ids = set()
    if os.path.exists(feature_file_path):
        try:
            df_existing = pl.read_parquet(feature_file_path)
            processed_ids = set(df_existing['comment_id'].to_list())
            logging.info(f"🔄 Resume: Found {len(processed_ids)} records already in output file.")
        except Exception as e:
            logging.warning(f"⚠️ Output file exists but couldn't be read: {e}")

    # B. Filter out processed records
    df_to_process = df_base.filter(~pl.col('comment_id').is_in(processed_ids))
    
    # C. Apply PILOT LIMIT (The only change in logic)
    if PILOT_MODE:
        if len(df_to_process) > PILOT_SIZE:
            logging.info(f"✂️ Cutting dataset to {PILOT_SIZE} records for Pilot test.")
            df_to_process = df_to_process.sample(n=PILOT_SIZE, seed=PILOT_SEED)
    
    total_to_process = len(df_to_process)
    if total_to_process == 0:
        logging.info("✅ No new records to process. Exiting.")
        return

    logging.info(f"⏳ Queue size: {total_to_process} records.")

    # 5. PROCESSING LOOP
    results_buffer = [] 
    count = 0
    
    for row in df_to_process.iter_rows(named=True):
        c_id = row['comment_id']
        text_input = row['text_content']
        
        score = -1 # Default error value
        
        try:
            # API Call
            llm_response = content_relevance_score(
                client=client,
                content=text_input,
                few_shot_examples=few_shot_examples
            )
            
            # Parse JSON
            data = json.loads(llm_response)
            score = data.get('content_relevance_score', -1)
            
            # Safety cast
            if score is None: score = -1
            else: score = int(score)

        except Exception as e:
            logging.warning(f"⚠️ Error processing ID {c_id}: {e}")
            score = -1
        
        # Add result to buffer
        results_buffer.append({
            "comment_id": c_id,
            "content_relevance_score": score
        })
        
        count += 1

        # 6. Incremental Saving (Batching)
        if count % BATCH_SAVE_SIZE == 0 or count == total_to_process:
            logging.info(f"💾 Saving batch... ({count}/{total_to_process})")
            
            df_new_chunk = pl.DataFrame(results_buffer)
            
            # Append Logic
            if os.path.exists(feature_file_path):
                try:
                    df_current = pl.read_parquet(feature_file_path)
                    # Vertical concat
                    df_combined = pl.concat([df_current, df_new_chunk])
                    df_combined.write_parquet(feature_file_path)
                except Exception as e:
                    logging.error(f"❌ Error saving batch: {e}")
            else:
                # Create new file
                df_new_chunk.write_parquet(feature_file_path)
            
            # Clear buffer
            results_buffer = []

    logging.info("✅ Generation Process Completed.")

if __name__ == "__main__":
    main()