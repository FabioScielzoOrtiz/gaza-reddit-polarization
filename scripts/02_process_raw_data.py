import os
import polars as pl
import logging 
import time

# --- LOGGING CONFIGURATION ---
# Set up basic configuration to log INFO level messages.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.info("⚙️ Starting data processing script.")

# --- PATH SETUP ---
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
raw_data_dir = os.path.join(project_path, 'data', 'raw_data')
processed_data_dir = os.path.join(project_path, 'data', 'processed_data')
os.makedirs(processed_data_dir, exist_ok=True) # Create output directory

start_time = time.time()

# --- DATA LOADING ---
logging.info("Loading raw Parquet data files...")
raw_data = {}
for raw_data_name in os.listdir(raw_data_dir):
    raw_data_path = os.path.join(raw_data_dir, raw_data_name)
    raw_data[raw_data_name.split('.')[0]] = pl.read_parquet(raw_data_path)

# Separate and vertically concatenate raw posts and comments
post_raw_data_list = [raw_data[k] for k in raw_data.keys() if 'posts' in k]
comments_raw_data_list = [raw_data[k] for k in raw_data.keys() if 'comments' in k]
post_raw_data = pl.concat(post_raw_data_list, how='vertical')
comments_raw_data = pl.concat(comments_raw_data_list, how='vertical')


# --- DATA PROCESSING AND CLEANING ---

logging.info("Starting data processing...")
# Unification: INNER JOIN comments with posts on 'post_id' to add context.
processed_data = comments_raw_data.join(post_raw_data, on='post_id', how='inner')

# Filter 1: Remove rows where comment_body is empty, [deleted], or [removed] (noise).
processed_data = processed_data.filter(
   ~pl.col('comment_body').is_in(["", "[deleted]", "[removed]"])
)

# Filter 2: Remove rows where both post title and body are noise (robustness check).
processed_data = processed_data.filter(
   ~ (
   (pl.col('post_title').is_in(["", "[deleted]", "[removed]"])) &
   (pl.col('post_body').is_in(["", "[deleted]", "[removed]"]))
   )
) 

# --- FEATURE GENERATION ---

# Create unified text variable ('text_content') for LLM input.
# Ensure post fields handle nulls/empties safely with .fill_null("").
processed_data = processed_data.with_columns((
    'Post Title:' + '\n\n' + 
    pl.col('post_title').fill_null("") + '\n\n' + 
    'Post Body:' + '\n\n' +
    pl.col('post_body').fill_null("") + '\n\n' + 
    'Comment Body:' + '\n\n' + 
    pl.col('comment_body')
 ).alias('text_content')
)

# --- SAVE OUTPUT ---

# Save the processed DataFrame to a single Parquet file.
processed_data_path = os.path.join(processed_data_dir, '02_processed_data.parquet')
processed_data.write_parquet(processed_data_path)

end_time = time.time()
elapsed_time = round((end_time - start_time) / 60, 2)
logging.info(f"Process completed in {elapsed_time} minutes. Final data saved to {processed_data_path}")