import os, sys
import json
import polars as pl
import logging
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score

# --- PATH CONFIGURATION ---
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
sys.path.insert(0, project_path)

# --- CONFIGURATION ---
from config.config_03b import (
    VALIDATION_THRESHOLD, 
    RELEVANCE_CUTOFF, 
)

# Directories
labeling_dir = os.path.join(project_path, 'data', 'labeled_samples')
train_sample_path = os.path.join(labeling_dir, '03a_train_sample_relevance.json')
val_sample_path = os.path.join(labeling_dir, '03a_val_sample_relevance.json')
reports_dir = os.path.join(project_path, 'data', 'validation_reports')
os.makedirs(reports_dir, exist_ok=True)

# Import LLM function (Ensure utils is updated)
from src.feature_engineering_utils import content_relevance_score

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
load_dotenv()

# --- REPORTING CLASS ---
class ValidationLogger:
    """Handles logging to both console and text file."""
    def __init__(self, feature_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(reports_dir, f"val_report_{feature_name}_{timestamp}.txt")
        self.buffer = []
        self.log(f"VALIDATION REPORT: {feature_name.upper()} - {timestamp}")
        self.log("="*60 + "\n")

    def log(self, message):
        print(message)
        self.buffer.append(str(message))

    def save(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.buffer))
        print(f"\n✅ Report saved to: {self.filename}")

# --- HELPER FUNCTIONS ---

def process_labeled_sample(file_path):
    """Loads manual JSON, reconstructs 'text_content', and filters nulls."""
    if not os.path.exists(file_path):
        logging.error(f"❌ File not found: {file_path}. Run script 03 first.")
        exit()

    try:
        # Read json as dict
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data: return pl.DataFrame([])
        
        # Read dict as df
        df = pl.DataFrame(data)
        
        # Filter rows without manual label
        if 'content_relevance_score' in df.columns:
             df = df.filter(pl.col('content_relevance_score').is_not_null())
             df = df.with_columns(pl.col('content_relevance_score').cast(pl.Int64))

        return df
    except Exception as e:
        logging.error(f"❌ Error loading {file_path}: {e}")
        exit()

def process_train_data_for_llm(df_train):
    """Converts the training DF to the list format for utils."""
    formatted_list = []
    for row in df_train.iter_rows(named=True):
        formatted_list.append({
            "text_content": row['text_content'], # Key used by your utils
            "content_relevance_score": row['content_relevance_score']
        })
    return formatted_list

# --- MAIN EXECUTION ---

def main():
    logger = ValidationLogger("content_relevance")
    logger.log("🚀 STARTING VALIDATION PROCESS")

    # 1. Load Data
    df_train = process_labeled_sample(train_sample_path)
    df_val = process_labeled_sample(val_sample_path)
    
    if len(df_train) == 0 or len(df_val) == 0:
        logger.log("❌ Error: Missing labeled data. Check 'data/labeled_samples' folder.")
        return

    logger.log(f"📂 Data Loaded -> Train (Few-Shot): {len(df_train)} | Val (Test): {len(df_val)}")

    # 2. Prepare Few-Shot Examples
    few_shot_examples = process_train_data_for_llm(df_train)

    # 3. Initialize OpenAI
    try:
        openai_client = OpenAI()
    except Exception as e:
        logger.log(f"❌ OpenAI Client Error: {e}")
        return

    # 4. Inference
    y_true = []
    y_pred = []
    
    logger.log(f"⏳ Running predictions on {len(df_val)} records...")

    for i, row in enumerate(df_val.iter_rows(named=True)):
        text_input = row['text_content']
        true_score = row['content_relevance_score']
        
        try:
            # CALL TO LLM (Using your utils)
            llm_response = content_relevance_score(
                client=openai_client, 
                content=text_input, 
                few_shot_examples=few_shot_examples
            )
            
            response_json = json.loads(llm_response)
            predicted_score = response_json.get('content_relevance_score')
            
            if predicted_score is None: predicted_score = -1
            else: predicted_score = int(predicted_score)

        except Exception as e:
            logging.warning(f"⚠️ Error in record {i}: {e}")
            predicted_score = -1

        y_true.append(true_score)
        y_pred.append(predicted_score)
        
        if (i+1) % 10 == 0: print(f"   Processed {i+1}/{len(df_val)}...")

    # 5. Metrics Calculation
    logger.log("-" * 60)
       
    # Binary Accuracy (Filter)
    # Convert to 1 (Relevant) or 0 (Irrelevant)
    y_true_bin = [1 if x >= RELEVANCE_CUTOFF else 0 for x in y_true]
    y_pred_bin = [1 if x >= RELEVANCE_CUTOFF else 0 for x in y_pred]
    binary_acc = accuracy_score(y_true_bin, y_pred_bin)
    
    logger.log(f"⚖️ Binary Filter Accuracy: {binary_acc:.2%} (Target: {VALIDATION_THRESHOLD:.0%})")
    logger.log(f"   (Threshold used: Score >= {RELEVANCE_CUTOFF} is Relevant)")

    # 6. Verdict
    logger.log("=" * 60)
    if binary_acc >= VALIDATION_THRESHOLD:
        logger.log("🚀 SUCCESS: The model is reliable for filtering content.")
        logger.log("👉 You can proceed to run '03c_generate_relevance_feature.py'")
    else:
        logger.log("🛑 FAILURE: The model filters incorrectly the data.")
        logger.log("👉 Action: Review your '03a_train_sample_relevance.json' examples or improve the LLM Prompt in content_relevance_score.")

    logger.save()

if __name__ == "__main__":
    main()