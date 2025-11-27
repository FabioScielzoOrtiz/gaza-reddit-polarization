import os, sys
import json
import polars as pl
import logging
import numpy as np
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error

# --- 1. CONFIGURATION & THRESHOLDS ---

# Quality Thresholds
ACCURACY_THRESHOLD = 0.80      # Target for Exact Match (Categorical/Ordinal)
ADJACENT_THRESHOLD = 0.90      # Target for Adjacent Match (Ordinal +/- 1)
MAE_THRESHOLD = 0.25           # Target for Continuous Error (Sentiment). Lower is better.

# Paths
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
sys.path.append(project_path)

# Data Directories
labeling_dir = os.path.join(project_path, 'data', 'manual_labeling')
train_sample_path = os.path.join(labeling_dir, '01_train_sample.json')
val_sample_path = os.path.join(labeling_dir, '02_val_sample.json')

# Output Report Directory
reports_dir = os.path.join(project_path, 'data', 'validation_reports')
os.makedirs(reports_dir, exist_ok=True)

# Import Scoring Functions from Utils
# NOTE: Ensure these functions are uncommented in your utils file
from src.feature_engineering_utils import (
    content_relevance_score,
    political_stance_score,
    discourse_tone_score,
    dominant_frame_score,
    argument_quality_score,
    sentiment_score_func  # Ensure this exists in utils!
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
load_dotenv()

# --- 2. REPORTING SYSTEM ---

class ValidationLogger:
    """Handles logging to both console and a text file report."""
    def __init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(reports_dir, f"validation_report_{timestamp}.txt")
        self.buffer = []
        self.log(f"VALIDATION REPORT - GENERATED AT {timestamp}")
        self.log("="*60 + "\n")

    def log(self, message):
        """Prints to console and appends to file buffer."""
        print(message) # Console output
        self.buffer.append(str(message))

    def save(self):
        """Writes the buffer to the .txt file."""
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                f.write("\n".join(self.buffer))
            print(f"\n✅ Report saved successfully to: {self.filename}")
        except Exception as e:
            logging.error(f"Failed to save report: {e}")

# Initialize Logger
report = ValidationLogger()

# --- 3. FEATURE CONFIGURATION MAP ---

FEATURE_CONFIG = {
    'content_relevance_score': {
        'func': content_relevance_score,
        'type': 'ordinal', # 0-5
        'cutoff': 3        # For binary filtering check
    },
    'political_stance': {
        'func': political_stance_score,
        'type': 'ordinal'  # 1-5
    },
    'argument_quality_score': {
        'func': argument_quality_score,
        'type': 'ordinal'  # 0-5
    },
    'sentiment_score': {
        'func': sentiment_score_func,
        'type': 'continuous' # Float -1.0 to 1.0
    },
    'discourse_tone': {
        'func': discourse_tone_score,
        'type': 'categorical' # Nominal (String)
    },
    'dominant_frame': {
        'func': dominant_frame_score,
        'type': 'categorical' # Nominal (String)
    }
}

# SELECT WHICH FEATURES TO VALIDATE IN THIS RUN
FEATURES_TO_VALIDATE = [
    'content_relevance_score',
    # 'political_stance',
    # 'sentiment_score',    # Uncomment to validate sentiment
    # 'discourse_tone',
    # 'dominant_frame', 
    # 'argument_quality_score'
]

# --- 4. HELPER FUNCTIONS ---

def load_and_prep_data(filepath, file_type):
    """Loads JSON, creates 'text_content', and filters unlabeled rows."""
    if not os.path.exists(filepath):
        logging.error(f"❌ File not found: {filepath}")
        exit()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not data: return pl.DataFrame([])

        df = pl.DataFrame(data)
        
        # Reconstruct text context
        df = df.with_columns((
            'Post Title:' + '\n\n' + pl.col('post_title').fill_null("") + '\n\n' + 
            'Post Body:' + '\n\n' + pl.col('post_body').fill_null("") + '\n\n' + 
            'Comment Body:' + '\n\n' + pl.col('comment_body')
        ).alias('text_content'))
        
        return df
    except Exception as e:
        logging.error(f"❌ Error loading {file_type}: {e}")
        exit()

def format_few_shot(df_train, feature_name):
    """Formats training examples specifically for the current feature."""
    formatted_list = []
    # Only pick rows where the specific feature is labeled
    df_filtered = df_train.filter(pl.col(feature_name).is_not_null())
    
    for row in df_filtered.iter_rows(named=True):
        formatted_list.append({
            "text": row['text_content'],
            "score": row[feature_name]
        })
    return formatted_list

def calculate_ordinal_metrics(y_true, y_pred):
    """Calculates metrics for numbered scales (1-5)."""
    exact_acc = accuracy_score(y_true, y_pred)
    # Adjacent Tolerance (+/- 1)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    diff = np.abs(y_true_arr - y_pred_arr)
    adj_acc = np.mean(diff <= 1)
    return exact_acc, adj_acc

def run_validation_for_feature(feature_name, client, df_train, df_val):
    """Core logic to run validation for a single feature."""
    
    config = FEATURE_CONFIG.get(feature_name)
    if not config:
        report.log(f"❌ Configuration not found for {feature_name}")
        return

    report.log(f"\n🔵 VALIDATING FEATURE: {feature_name.upper()} ({config['type']})")
    
    # 1. Prepare Data
    train_examples = format_few_shot(df_train, feature_name)
    
    # Filter validation set for rows labeled for THIS feature
    df_val_clean = df_val.filter(pl.col(feature_name).is_not_null())
    
    if len(df_val_clean) == 0:
        report.log(f"⚠️ No validation labels found for {feature_name}. Skipping.")
        return

    # 2. Inference Loop
    y_true = []
    y_pred = []
    
    report.log(f"   Running predictions on {len(df_val_clean)} records...")
    
    for i, row in enumerate(df_val_clean.iter_rows(named=True)):
        text_input = row['text_content']
        actual_val = row[feature_name]
        
        try:
            # Dynamic Function Call
            response_str = config['func'](
                client=client,
                predict_data=text_input,
                train_data=train_examples
            )
            response_json = json.loads(response_str)
            predicted_val = response_json.get(feature_name)
            
            # Type Casting based on config
            if config['type'] == 'ordinal':
                predicted_val = int(predicted_val) if predicted_val is not None else -1
                actual_val = int(actual_val)
            
            elif config['type'] == 'continuous':
                # Float conversion for Sentiment
                predicted_val = float(predicted_val) if predicted_val is not None else 0.0
                actual_val = float(actual_val)

            else:
                # Categorical (String)
                predicted_val = str(predicted_val) if predicted_val is not None else "ERROR"
                actual_val = str(actual_val)

        except Exception as e:
            logging.warning(f"   ⚠️ Error record {i}: {e}")
            if config['type'] == 'ordinal': predicted_val = -1
            elif config['type'] == 'continuous': predicted_val = 0.0
            else: predicted_val = "ERROR"

        y_true.append(actual_val)
        y_pred.append(predicted_val)

    # 3. Metrics & Reporting
    report.log("-" * 60)
    report.log(f"📊 METRICS REPORT: {feature_name}")
    
    # --- ORDINAL LOGIC ---
    if config['type'] == 'ordinal':
        exact, adjacent = calculate_ordinal_metrics(y_true, y_pred)
        report.log(f"   🎯 Exact Accuracy:     {exact:.2%} (Target: {ACCURACY_THRESHOLD:.0%})")
        report.log(f"   ok Adjacent Accuracy:  {adjacent:.2%} (Target: {ADJACENT_THRESHOLD:.0%}) (Tolerance +/- 1)")
        
        if feature_name == 'content_relevance_score':
            cutoff = config['cutoff']
            bin_true = [1 if x >= cutoff else 0 for x in y_true]
            bin_pred = [1 if x >= cutoff else 0 for x in y_pred]
            bin_acc = accuracy_score(bin_true, bin_pred)
            report.log(f"   ⚖️ Binary Filter Acc:  {bin_acc:.2%} (Score >= {cutoff})")

        report.log("\n   Confusion Matrix:\n" + str(confusion_matrix(y_true, y_pred)))

    # --- CONTINUOUS LOGIC (SENTIMENT) ---
    elif config['type'] == 'continuous':
        mae = mean_absolute_error(y_true, y_pred)
        report.log(f"   📉 Mean Absolute Error (MAE): {mae:.4f} (Target: < {MAE_THRESHOLD})")
        
        # Optional: Polarity Check (Did we get the sign right?)
        same_sign = np.mean(np.sign(y_true) == np.sign(y_pred))
        report.log(f"   ➕➖ Polarity Agreement:       {same_sign:.2%}")
        
        if mae <= MAE_THRESHOLD:
            report.log("   ✅ SUCCESS: Error is within acceptable limits.")
        else:
            report.log("   🛑 FAILURE: High error rate.")

    # --- CATEGORICAL LOGIC ---
    elif config['type'] == 'categorical':
        acc = accuracy_score(y_true, y_pred)
        report.log(f"   🎯 Exact Accuracy:     {acc:.2%} (Target: {ACCURACY_THRESHOLD:.0%})")
        report.log("\n   Detailed Report:\n" + classification_report(y_true, y_pred, zero_division=0))
        report.log("\n   Confusion Matrix:\n" + str(confusion_matrix(y_true, y_pred)))

    report.log("-" * 60)


# --- 5. MAIN EXECUTION ---

def main():
    report.log("🚀 STARTING MULTI-FEATURE VALIDATION")
    
    # Load Data
    df_train = load_and_prep_data(train_sample_path, "Train")
    df_val = load_and_prep_data(val_sample_path, "Validation")
    
    # Init Client
    try:
        openai_client = OpenAI()
    except Exception:
        logging.error("❌ OpenAI Client failed.")
        exit()

    # Iterate over selected features
    for feature in FEATURES_TO_VALIDATE:
        run_validation_for_feature(feature, openai_client, df_train, df_val)

    # Save final report
    report.save()

if __name__ == "__main__":
    main()