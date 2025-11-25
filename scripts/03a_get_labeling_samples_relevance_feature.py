import os, sys
import json
import logging 
import polars as pl
import math

# --- PATH CONFIGURATION ---
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
sys.path.insert(0, project_path)

from config.config_03a import (
    SAMPLE_N, # Total target number of samples (Manual + Random) 
    SAMPLE_SEED, # Seed for reproducibility
    VAL_SAMPLE_RATIO, # 80% for Blind Validation, 20% for Few-Shot Training
    # Add specific comment_ids here to FORCE them into the specific set. Useful for including known edge cases (sarcasm, short text) in the prompt.
    MANUAL_TRAIN_IDS, 
    MANUAL_VAL_IDS 
)

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Input: Base processed data (from Step 02)
base_data_path = os.path.join(project_path, 'data', 'processed_data', '02_processed_data.parquet')

# Output: New dedicated folder for manual labeling inputs
labeling_dir = os.path.join(project_path, 'data', 'labeled_samples')
os.makedirs(labeling_dir, exist_ok=True)

train_sample_path = os.path.join(labeling_dir, '03a_train_sample_relevance.json')
val_sample_path = os.path.join(labeling_dir, '03a_val_sample_relevance.json')

if os.path.exists(train_sample_path) and os.path.exists(val_sample_path):
    logging.info("⛔ Train and validation samples already exist --> Process Stopped")
    exit()

def main():
    
    logging.info("⚙️ Starting generation of samples (Manual + Random)...")

    # 1. LOAD DATA
    try:
        df_base = pl.read_parquet(base_data_path)
        logging.info(f"📂 Base dataset loaded: {len(df_base)} records.")
    except Exception as e:
        logging.error(f"❌ Error loading data: {e}")
        exit()

    # 2. EXTRACT MANUAL SAMPLES
    # Identify manual rows
    df_manual_train = df_base.filter(pl.col('comment_id').is_in(MANUAL_TRAIN_IDS))
    df_manual_val = df_base.filter(pl.col('comment_id').is_in(MANUAL_VAL_IDS))
    
    # Check if we found all requested IDs
    if len(df_manual_train) < len(MANUAL_TRAIN_IDS):
        found = df_manual_train['comment_id'].to_list()
        missing = set(MANUAL_TRAIN_IDS) - set(found)
        logging.warning(f"⚠️ Some MANUAL TRAIN IDs were not found in dataset: {missing}")

    if len(df_manual_val) < len(MANUAL_VAL_IDS):
        found = df_manual_val['comment_id'].to_list()
        missing = set(MANUAL_VAL_IDS) - set(found)
        logging.warning(f"⚠️ Some MANUAL VAL IDs were not found in dataset: {missing}")

    logging.info(f"🔧 Manual Samples Extracted -> Train: {len(df_manual_train)} | Val: {len(df_manual_val)}")

    # 3. PREPARE RANDOM POOL (Excluding Manual IDs to avoid duplicates/leakage)
    all_manual_ids = MANUAL_TRAIN_IDS + MANUAL_VAL_IDS
    df_pool = df_base.filter(~ pl.col('comment_id').is_in(all_manual_ids))
    
    # 4. CALCULATE QUOTAS
    val_n = int(SAMPLE_N * VAL_SAMPLE_RATIO)
    train_n = SAMPLE_N - val_n   
    manual_val_n = len(df_manual_val)
    manual_train_n = len(df_manual_train)
    
    # Calculate how many randoms we still need
    random_train_n = max(0, train_n - manual_train_n)
    random_val_n = max(0, val_n - manual_val_n)
    random_total_n = random_train_n + random_val_n
    logging.info(f"🎲 Random Samples Needed -> Train: {random_train_n} | Val: {random_val_n}")

    # 5. SAMPLE RANDOM DATA
    try:
        df_random_selected = df_pool.sample(n=random_total_n, seed=SAMPLE_SEED, with_replacement=False)
    except Exception:
        logging.warning("⚠️ Pool is smaller than requested samples. Taking everything available.")
        df_random_selected = df_pool

    # Split the random selection into train and val chunks
    df_random_train = df_random_selected[:random_train_n]
    df_random_val = df_random_selected[random_train_n:]

    # 6. COMBINE MANUAL + RANDOM
    df_train_final = pl.concat([df_manual_train, df_random_train])
    df_val_final = pl.concat([df_manual_val, df_random_val])

    logging.info(f"📊 FINAL SPLIT -> Train (Few-Shot): {len(df_train_final)} | Validation (Blind): {len(df_val_final)}")

    # 7. EXPORT TO JSON
    data_columns_to_show = ['comment_id', 'post_title', 'post_body', 'comment_body', 'text_content']
    features_to_label = [
        'content_relevance_score', 
        #'political_stance', 
        #'sentiment_score', 
        #'discourse_tone', 
        #'dominant_frame', 
        #'argument_quality_score'
        # 'sentiment_score'
    ]

    def export_to_json(df, file_path):
        export_list = []
        for row in df.iter_rows(named=True):
            item = {k: row[k] for k in data_columns_to_show if k in row}
            item.update({k: None for k in features_to_label})
            export_list.append(item)        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_list, f, indent=4, ensure_ascii=False)
        logging.info(f"💾 File saved: {file_path}")

    export_to_json(df_train_final, train_sample_path)
    export_to_json(df_val_final, val_sample_path)

    logging.info("✅ Process completed. Check 'data/labeled_samples' folder.")

if __name__ == "__main__":
    main()