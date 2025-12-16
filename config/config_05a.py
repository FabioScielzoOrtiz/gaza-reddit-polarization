# Nombre del modelo de Embeddings (Coste muy bajo y alto rendimiento)
EMBEDDINGS_MODEL = "text-embedding-3-small" 

# Tamaño del lote para enviar a la API (OpenAI acepta arrays de inputs)
# 100-500 es un buen rango.
BATCH_SIZE = 250 

MAX_CONCURRENT_REQUESTS = 50 # CONCURRENCY CONTROL (Embeddings tiene rate limits altos, 50 es seguro)

PILOT_MODE = True # Set to True to process only a small sample (e.g., 10 records). Set to False to process the entire dataset.
PILOT_SIZE = 25
PILOT_SEED = 111