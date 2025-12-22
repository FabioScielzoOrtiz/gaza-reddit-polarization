SAMPLE_N = 10 # 100 # Total target number of samples (Manual + Random)
SAMPLE_SEED = 123 # Seed for reproducibility
VAL_SAMPLE_RATIO = 0.80 # Proportion of labeling samples for validation. Remaining for training (few-shot-llm)
# Add specific comment_ids here to FORCE them into the specific set. Useful for including known edge cases (sarcasm, short text) in the prompt.
MANUAL_TRAIN_IDS = [ 
    # 'msmtwuf', 
    # 'comment_id_X1',
    # 'comment_id_X2',
]
MANUAL_VAL_IDS = [
    # 'msjyrcc',
    # 'comment_id_Y1',
    # 'comment_id_Y2',
]
DATA_COLUMNS_TO_INCLUDE = [ # Data columns to include in the labeling samples
    'comment_id', 
    'post_title', 
    'post_body', 
    'comment_body', 
    'text_content'
]
