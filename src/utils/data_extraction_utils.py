import glob
import re
import os
import polars as pl
import praw
import prawcore
import datetime as dt
import logging
import itertools
import time

# --- AUTHENTICATION ---
def authenticate_praw(CLIENT_ID, CLIENT_SECRET, USER_AGENT):
    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT,
        )
        reddit.user.me() 
        logging.info("✅ PRAW initialized successfully.")
        return reddit
    except Exception as e:
        logging.error(f"❌ Authentication failed: {e}")
        return None

# --- GESTIÓN DE ESTADO Y ARCHIVOS ---

def _load_state_from_folder(folder_path):
    """
    Escanea la carpeta caché para ver qué IDs ya tenemos guardados
    y determinar el número del siguiente lote.
    """
    post_ids_seen = set()
    max_batch_num = 0
    
    if not os.path.exists(folder_path):
        return post_ids_seen, 1

    logging.info(f"🔄 Scanning cache in: {folder_path}...")

    # Buscamos archivos de posts para sacar los IDs ya procesados
    post_files = glob.glob(os.path.join(folder_path, "batch_posts_*.parquet"))
    
    if post_files:
        try:
            # Leemos solo la columna post_id para ir rápido
            df_temp = pl.read_parquet(post_files, columns=['post_id'])
            post_ids_seen = set(df_temp['post_id'].to_list())
        except Exception as e:
            logging.warning(f"⚠️ Error reading cache files: {e}")

    # Calculamos el siguiente número de batch (batch_posts_1.parquet -> 1)
    for f in post_files:
        match = re.search(r'batch_posts_(\d+)\.parquet', f)
        if match:
            num = int(match.group(1))
            if num > max_batch_num:
                max_batch_num = num
    
    next_batch = max_batch_num + 1
    
    if post_ids_seen:
        logging.info(f"⏩ Resuming: Found {len(post_ids_seen)} posts already saved. Starting Batch {next_batch}.")
    else:
        logging.info("🆕 No cache found. Starting from scratch.")

    return post_ids_seen, next_batch

def _save_buffers_to_disk(post_buffer, comment_buffer, output_dir, batch_num):
    """Guarda lo que haya en memoria a disco inmediatamente."""
    
    # 1. Guardar Posts
    if post_buffer:
        try:
            df_p = pl.DataFrame(post_buffer)
            p_path = os.path.join(output_dir, f"batch_posts_{batch_num}.parquet")
            df_p.write_parquet(p_path)
            logging.info(f"💾 Saved Batch {batch_num}: {len(df_p)} Posts -> {p_path}")
        except Exception as e:
            logging.error(f"❌ Error saving Posts Batch {batch_num}: {e}")

    # 2. Guardar Comentarios (asociados a esos posts)
    if comment_buffer:
        try:
            df_c = pl.DataFrame(comment_buffer)
            c_path = os.path.join(output_dir, f"batch_comments_{batch_num}.parquet")
            df_c.write_parquet(c_path)
            logging.info(f"💾 Saved Batch {batch_num}: {len(df_c)} Comments -> {c_path}")
        except Exception as e:
            logging.error(f"❌ Error saving Comments Batch {batch_num}: {e}")

# --- FUNCIÓN PRINCIPAL DE EXTRACCIÓN ---

def run_data_extraction(reddit, subreddits, queries, sorts, max_limit, time_filter, output_dir, batch_size):
    
    # 1. Cargar estado previo (si existe) para no repetir
    post_ids_seen, batch_counter = _load_state_from_folder(output_dir)
    
    # Buffers en memoria (se vacían cada vez que guardamos un lote)
    post_buffer = []
    comment_buffer = []
    
    extraction_time_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    extraction_combinations = list(itertools.product(subreddits, queries, sorts))

    logging.info(f"🚀 Starting extraction loop. Batch size: {batch_size} posts.")

    for subreddit_name, query, sort in extraction_combinations:
        logging.info(f"📂 Processing: r/{subreddit_name} | Q: '{query}' | Sort: {sort}")

        try:
            search_results = reddit.subreddit(subreddit_name).search(
                query, sort=sort, time_filter=time_filter, limit=max_limit
            )
            
            for post in search_results:
                
                # --- A. CHECKPOINT: ¿Ya tenemos este post? ---
                if post.id in post_ids_seen:
                    continue # Saltamos y pasamos al siguiente
                
                logging.info(f"⚙️ Processing Post: {post.id}")

                # --- B. EXTRAER POST (Memoria) ---
                post_record = {
                    'post_id': post.id,
                    'post_subreddit': post.subreddit.display_name,
                    'post_title': post.title,
                    'post_body': post.selftext,
                    'post_score': post.score,
                    'post_upvote_ratio': post.upvote_ratio,
                    'post_num_comments': post.num_comments,
                    'post_created_utc': post.created_utc,
                    'post_created_date': dt.datetime.fromtimestamp(post.created_utc, dt.timezone.utc).isoformat(),
                    'post_url': post.url,
                    'extraction_query': query,
                    'extraction_sort': sort,
                    'extraction_time': extraction_time_utc,
                }
                post_buffer.append(post_record)
                post_ids_seen.add(post.id) # Lo marcamos como visto

                # --- C. EXTRAER COMENTARIOS DEL POST (Inmediatamente) ---
                # Esto asegura el flujo "Post -> Comentarios"
                try:
                    post.comments.replace_more(limit=0)
                    comments_found = 0
                    
                    for comment in post.comments:
                        comment_record = {
                            'comment_id': comment.id,
                            'post_id': post.id, # Link clave
                            'comment_body': comment.body,
                            'comment_score': comment.score,
                            'comment_created_utc': comment.created_utc,
                            'comment_created_date': dt.datetime.fromtimestamp(comment.created_utc, dt.timezone.utc).isoformat(),
                        }
                        comment_buffer.append(comment_record)
                        comments_found += 1
                    
                    logging.info(f"   ↳ Collected {comments_found} comments for Post {post.id}")

                except Exception as e:
                    logging.warning(f"   ⚠️ Error getting comments for {post.id}: {e}")

                # --- D. GUARDADO INCREMENTAL (BATCH) ---
                # Si hemos acumulado suficientes posts (ej: 20), volcamos a disco.
                if len(post_buffer) >= batch_size:
                    logging.info("📥 Batch limit reached. Writing to disk...")
                    _save_buffers_to_disk(post_buffer, comment_buffer, output_dir, batch_counter)
                    
                    # Limpiar buffers y avanzar contador
                    post_buffer = []
                    comment_buffer = []
                    batch_counter += 1
            
        except Exception as e:
            logging.error(f"❌ Error in search loop: {e}")
            time.sleep(2)

    # --- GUARDADO FINAL ---
    # Si quedaron datos en el buffer al terminar todo, los guardamos
    if post_buffer or comment_buffer:
        logging.info("📥 Saving final residual data...")
        _save_buffers_to_disk(post_buffer, comment_buffer, output_dir, batch_counter)

    logging.info("✅ Extraction loop finished.")