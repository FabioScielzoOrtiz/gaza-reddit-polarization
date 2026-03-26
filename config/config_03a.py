VAL_N = 50 # Total target number of samples (Manual + Random)
SAMPLE_SEED = 123 # Seed for reproducibility
# Add specific comment_ids here to FORCE them into the specific set. Useful for including known edge cases (sarcasm, short text) in the prompt.
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
