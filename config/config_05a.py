# Nombre del modelo de Embeddings (Coste muy bajo y alto rendimiento)
EMBEDDING_MODEL = "text-embedding-3-small" 

# Tamaño del lote para enviar a la API (OpenAI acepta arrays de inputs)
# 100-500 es un buen rango.
BATCH_SIZE = 250 

MAX_CONCURRENT_REQUESTS = 50 # CONCURRENCY CONTROL (Embeddings tiene rate limits altos, 50 es seguro)