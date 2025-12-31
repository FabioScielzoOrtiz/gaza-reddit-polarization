import polars as pl
import datetime as dt
from dotenv import load_dotenv
import os
import logging
import sys
import glob
import shutil

# --- CONFIGURACIÓN DE RUTAS ---
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
sys.path.insert(0, project_path)

from config.config_01 import (
    LIST_SUBREDDITS, LIST_QUERIES, LIST_SORTS, 
    MAX_LIMIT, TIME_FILTER, BATCH_SIZE
)
from src.data_extraction_utils import authenticate_praw, run_data_extraction

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = "ResearchScript v2.0 BatchSaver"

def main():
    reddit = authenticate_praw(CLIENT_ID, CLIENT_SECRET, USER_AGENT)
    if not reddit: sys.exit(1)

    # 1. Definir carpetas
    final_output_dir = os.path.join(project_path, 'data', 'raw_data')
    os.makedirs(final_output_dir, exist_ok=True)
    
    # CARPETA CACHÉ FIJA: Aquí se guardan los trozos temporales.
    # Si el código falla y vuelves a ejecutar, mirará aquí para continuar.
    temp_dir = os.path.join(project_path, 'data', 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # 2. Ejecutar extracción (Guardado automático en cache_dir)
    # batch_size=10 significa que cada 10 posts, guarda un archivo. 
    # Si se rompe en el post 15, tendrás el archivo del 1 al 10 seguro en disco.
    run_data_extraction(
        reddit, 
        LIST_SUBREDDITS, LIST_QUERIES, LIST_SORTS, 
        MAX_LIMIT, TIME_FILTER, 
        output_dir=temp_dir, 
        batch_size=BATCH_SIZE  
    )

    # 3. Consolidación (Unir todos los trozos)
    logging.info("🔄 Consolidating batches into final file...")
    timestamp = dt.datetime.now().strftime('%Y%m%d%H%M%S')

    try:
        # Unir Posts
        post_files = glob.glob(os.path.join(temp_dir, "batch_posts_*.parquet"))
        if post_files:
            df_final_posts = pl.read_parquet(post_files)
            path_posts = os.path.join(final_output_dir, f"posts_raw_{timestamp}.parquet")
            df_final_posts.write_parquet(path_posts)
            logging.info(f"✅ Final Posts Saved: {path_posts} (Total: {len(df_final_posts)})")
        
        # Unir Comentarios
        comment_files = glob.glob(os.path.join(temp_dir, "batch_comments_*.parquet"))
        if comment_files:
            df_final_comments = pl.read_parquet(comment_files)
            path_comments = os.path.join(final_output_dir, f"comments_raw_{timestamp}.parquet")
            df_final_comments.write_parquet(path_comments)
            logging.info(f"✅ Final Comments Saved: {path_comments} (Total: {len(df_final_comments)})")

        # 4. Limpieza archvos temporales si el proceso finalizó correctamente
        shutil.rmtree(temp_dir)
        logging.info("🧹 Cache cleared.")

    except Exception as e:
        logging.error(f"❌ Error merging files: {e}")
        logging.warning(f"⚠️ Your partial data is still safe in {temp_dir}")

if __name__ == "__main__":
    main()