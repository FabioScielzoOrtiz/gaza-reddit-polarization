import polars as pl
import datetime as dt
from dotenv import load_dotenv
import os
import logging
import sys
import time

# ------------------------------------------------------------------------------------------
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
sys.path.insert(0, project_path)
# ------------------------------------------------------------------------------------------

from config.data_extraction_config import (
    LIST_SUBREDDITS, LIST_QUERIES, LIST_SORTS, 
    MAX_LIMIT, TIME_FILTER
)
# Se usa 'data_extraction_uitls' para coincidir con el nombre de archivo subido
from src.data_extraction_uitls import authenticate_praw, run_extraction 

# --- 0. CONFIGURATION & SETUP ---

# Dynamic data_extraction_id based on the current datetime (YYYYMMDDHHMMSS)
data_extraction_id = dt.datetime.now().strftime('%Y%m%d%H%M%S') 

# Define logging directories and file path
logs_dir = os.path.join(project_path, 'logs')
logs_file_name = f'data_extraction_{data_extraction_id}.log'
os.makedirs(logs_dir, exist_ok=True) 
logs_file_path = os.path.join(logs_dir, logs_file_name)

# Configure logging settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_file_path, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.info(f"Logging configured. Output file: {logs_file_path}")

# Load environment variables
load_dotenv(os.path.join(project_path, '.env'))

CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
USER_AGENT = "ResearchScript v1.0 by /u/Hour_Sell5070" 

# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    
    # 1. Authenticate and exit if failed 
    reddit = authenticate_praw(CLIENT_ID, CLIENT_SECRET, USER_AGENT)
    if not reddit:
        logging.error("❌ Authentication failed. Exiting script.")
        sys.exit(1)
        
    # 2. Run the full extraction process
    logging.info("Starting data extraction...")
    start_time = time.time()
    post_data_list, comment_data_list = run_extraction(
        reddit, 
        LIST_SUBREDDITS, 
        LIST_QUERIES, 
        LIST_SORTS,
        MAX_LIMIT,
        TIME_FILTER
    )
    end_time = time.time()
    logging.info(f"✅ Data Extraction completed in {round((end_time - start_time)/60, 2)} minutes.")

    # 3. Convert to Polars DataFrames and Save (RAW Data Only)
    
    logging.info("Saving extracted data")

    output_data_dir = os.path.join(project_path, 'data', 'raw_data')
    os.makedirs(output_data_dir, exist_ok=True)
    
    # --- POSTS (RAW) ---
    if post_data_list:
        df_posts = pl.DataFrame(post_data_list)
        output_filename_posts = f"posts_data_raw_{data_extraction_id}.parquet"
        output_file_path_posts = os.path.join(output_data_dir, output_filename_posts)
        df_posts.write_parquet(output_file_path_posts)
        logging.info(f"📁 RAW POSTS data saved successfully to {output_file_path_posts} (Records: {len(df_posts)})")
    else:
        logging.warning("❌ No post data was extracted.")
        
    # --- COMMENTS (RAW) ---
    if comment_data_list:
        df_comments = pl.DataFrame(comment_data_list)
        output_filename_comments = f"comments_data_raw_{data_extraction_id}.parquet"
        output_file_path_comments = os.path.join(output_data_dir, output_filename_comments)
        df_comments.write_parquet(output_file_path_comments)
        logging.info(f"📁 RAW COMMENTS data saved successfully to {output_file_path_comments} (Records: {len(df_comments)})")
    else:
        logging.warning("❌ No comment data was extracted.")
      
    logging.info(f"📥 DATA EXTRACTION COMPLETED")
    