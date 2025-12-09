#################################################################################################

# Imports 

import os
import json
import logging
import asyncio
import polars as pl
import numpy as np
from openai import AsyncOpenAI
from sklearn.metrics import accuracy_score, mean_absolute_error
from tqdm.asyncio import tqdm_asyncio 
from sklearn.decomposition import PCA

#################################################################################################

# Logging Configuration

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#################################################################################################

# ==============================================================================
# 1. CONTENT RELEVANCE SCORE (Filtrado) - ASYNC
# ==============================================================================

async def content_relevance_score(client: AsyncOpenAI, model_name: str = "gpt-4o-mini", temperature: float = 0.0, content: str = None, few_shot_examples: list = None):
    """
    Calcula la relevancia temática usando ejemplos Few-Shot dinámicos (Versión Async).
    """

    prompt = f"""
You are a content rating specialist for an academic study on public opinion regarding the Gaza conflict on Reddit.

Your task is to assign a numerical **Relevance Score** from **0 (Not Related)** to **5 (Directly Related)** to the provided text.

---
**STRICT GUIDELINE:**
1. **FOCUS:** The score MUST primarily reflect the relevance of the **Comment Body**. Use the Post Title/Body ONLY as context.
2. **CRITERIA:** Do NOT penalize based on tone, quality, or brevity. Only evaluate topical connection to the Israel-Palestine conflict.

---
**TOPICAL RELEVANCE SCALE (0-5):**
* **5 - Directly Related:** Explicit mention of the conflict, main actors (Israel, Hamas, IDF, Gaza), or core events.
* **4 - Clearly Related:** Brief mentions, strong reactions, or aggressive statements unambiguously about the conflict.
* **3 - Marginal Context:** Related keywords (Middle East, UN, War) without explicit ties to Gaza/Israel. Broad context.
* **2 - Accidental/Trivial:** Keywords used in non-political context (e.g., travel advice) or pure noise in a related thread.
* **1 - Off-Topic Noise:** Personal attacks or emotional outbursts unrelated to the topic.
* **0 - Discard/Spam:** Completely unrelated content.

---
**EXPERT KNOWLEDGE: REFERENCE SAMPLES (Ground Truth)**
Use the following expert-labeled examples as your calibration standard. You must align your scoring logic with these cases:

{few_shot_examples}

---
**OUTPUT FORMAT:**
Return a single JSON object. Example: {{"content_relevance_score": 4}}

**TEXT TO CLASSIFY:**
{content}
"""

    try:
        response = await client.chat.completions.create(
            model=model_name, 
            messages=[
                {"role": "system", "content": "You are a helpful classification assistant. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
            temperature=temperature 
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in OpenAI API call (Relevance): {e}")
        return json.dumps({"content_relevance_score": None})

#################################################################################################

# ==============================================================================
# 2. POLITICAL STANCE SCORE - ASYNC
# ==============================================================================

async def political_stance_score(client: AsyncOpenAI,  model_name: str = "gpt-4o-mini", temperature: float = 0.0, content: str = None, few_shot_examples: list = None):
    prompt = f"""
You are an expert political analyst for a study on the Gaza conflict.

Your task is to assign a **Political Stance Score** from **1 (Pro-Palestine)** to **5 (Pro-Israel)**.

---
**STRICT GUIDELINES:**
1. **FOCUS:** Analyze the **Comment Body**. Use Post context only for interpretation.
2. **TONE vs STANCE:** Distinguish between aggressive tone and political direction. An aggressive comment can be Pro-Israel (5) or Pro-Palestine (1).
3. **NEUTRALITY:** Score 3 is ONLY for truly balanced analysis or unrelated neutral facts.

---
**POLITICAL STANCE SCALE (1-5):**
* **5 - Strongly Pro-Israel:** Explicit support for Israel/IDF, justification of actions, condemnation of Hamas as sole aggressor.
* **4 - Leaning Pro-Israel:** Empathy for Israeli civilians, focus on security rights, mild criticism of Palestine.
* **3 - Neutral/Balanced:** Academic analysis, criticizing both sides equally, or factual reporting without opinion.
* **2 - Leaning Pro-Palestine:** Focus on humanitarian crisis in Gaza, criticism of Israeli policies, empathy for Palestinian civilians.
* **1 - Strongly Pro-Palestine:** Accusations of genocide/apartheid against Israel, strong support for Palestinian resistance.

---
**EXPERT KNOWLEDGE: REFERENCE SAMPLES (Ground Truth)**
Learn from these human-labeled examples to calibrate your judgment:

{few_shot_examples}

---
**OUTPUT FORMAT:**
Return a single JSON object. Example: {{"political_stance": 2}}

**TEXT TO CLASSIFY:**
{content}
"""
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a political analyst. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in OpenAI API call (Stance): {e}")
        return json.dumps({"political_stance": None})

#################################################################################################

# ==============================================================================
# 3. DISCOURSE TONE - ASYNC
# ==============================================================================

async def discourse_tone_score(client: AsyncOpenAI,  model_name: str = "gpt-4o-mini", temperature: float = 0.0, content: str = None, few_shot_examples: list = None):
    prompt = f"""
You are an expert linguist analyzing political discourse on Reddit regarding the Gaza conflict.

Your task is to identify the **Dominant Discourse Tone** of the provided comment.

---
**CATEGORIES (Choose exactly ONE):**

1. **Analytical:** Objective tone, uses logic, cites sources, focuses on facts or strategic analysis. Low emotional charge.
2. **Emotional:** Dominant expression of sadness, grief, fear, empathy, or despair. Focus on suffering (humanitarian).
3. **Hostile:** Aggressive, insulting, uses hate speech, dehumanization, or ad-hominem attacks against users or groups.
4. **Sarcastic:** Uses irony, mockery, or satire. Says the opposite of what is meant to ridicule a position.
5. **Informative:** Neutral sharing of links, breaking news, or clarifications without taking a clear analytical or emotional stance.
6. **Other:** Content that does not fit the above categories.

---
**EXPERT KNOWLEDGE: REFERENCE SAMPLES (Ground Truth)**
{few_shot_examples}

---
**OUTPUT FORMAT:**
Return a single JSON object with the exact category name.
Example: {{"discourse_tone": "Sarcastic"}}

**TEXT TO CLASSIFY:**
{content}
"""
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a linguist. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in Tone: {e}")
        return json.dumps({"discourse_tone": None})

#################################################################################################

# ==============================================================================
# 4. DOMINANT FRAME - ASYNC
# ==============================================================================

async def dominant_frame_score(client: AsyncOpenAI, model_name: str = "gpt-4o-mini", temperature: float = 0.0, content: str = None, few_shot_examples: list = None):
    prompt = f"""
You are a media analyst studying framing effects in the Gaza conflict.

Your task is to identify the **Dominant Frame** used in the text. This is the "lens" through which the user views the issue.

---
**CATEGORIES (Choose exactly ONE):**

1. **Humanitarian/Legal:** Focus on human rights, international law (ICJ/UN), war crimes, genocide definitions, civilian casualties, aid, and suffering.
2. **Security/Military:** Focus on military tactics, Hamas capabilities, IDF strategy, borders, hostages, terrorism, and self-defense rights.
3. **Geopolitical/Political:** Focus on international relations (US/Iran/Egypt), UN resolutions, domestic politics (Netanyahu/Biden), and diplomatic solutions.
4. **Media/Narrative:** Focus on how the war is reported, bias in news sources (CNN/BBC/Al Jazeera), propaganda ("hasbara"), or disinformation.
5. **Historical/Religious:** Focus on historical claims (1948, 1967), biblical/religious justifications, or long-term historical context.
6. **Other:** Content that does not fit the above frames.

---
**EXPERT KNOWLEDGE: REFERENCE SAMPLES (Ground Truth)**
{few_shot_examples}

---
**OUTPUT FORMAT:**
Return a single JSON object with the exact category name.
Example: {{"dominant_frame": "Security/Military"}}

**TEXT TO CLASSIFY:**
{content}
"""
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a media analyst. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in Frame: {e}")
        return json.dumps({"dominant_frame": None})

#################################################################################################

# ==============================================================================
# 5. ARGUMENT QUALITY SCORE - ASYNC
# ==============================================================================

async def argument_quality_score(client: AsyncOpenAI, model_name: str = "gpt-4o-mini", temperature: float = 0.0, content: str = None, few_shot_examples: list = None):
    prompt = f"""
You are an academic researcher evaluating the quality of public deliberation about Gaza conflict.

Your task is to assign an **Argument Quality Score** from **0 to 5** based on the sophistication and justification of the text.

---
**SCORING RUBRIC:**

* **0 - Spam/Non-Argument:** Broken text, bots, or completely unintelligible noise.
* **1 - Low (Pure Reaction):** Name-calling, simple slogans ("Free Palestine", "I stand with Israel"), or single-word emotional reactions without reasons.
* **2 - Basic (Opinion):** Stating a clear position but with minimal or weak justification. Repetitive talking points.
* **3 - Moderate (Justified Opinion):** A position supported by at least one coherent reason or personal anecdote. Clear logic but limited depth.
* **4 - High (Reasoned Argument):** Well-structured argument linking evidence to claims. Shows nuance or acknowledges context.
* **5 - Elite (Sophisticated Discourse):** Exceptional depth. Cites specific sources/laws, considers counter-arguments, or synthesizes complex information.

---
**EXPERT KNOWLEDGE: REFERENCE SAMPLES (Ground Truth)**
{few_shot_examples}

---
**OUTPUT FORMAT:**
Return a single JSON object.
Example: {{"argument_quality_score": 3}}

**TEXT TO CLASSIFY:**
{content}
"""
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a researcher. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in Quality: {e}")
        return json.dumps({"argument_quality_score": None})

#################################################################################################

# ==============================================================================
# 5. SENTIMENT SCORE - ASYNC
# ==============================================================================

async def sentiment_score(client: AsyncOpenAI, model_name: str = "gpt-4o-mini", temperature: float = 0.0, content: str = None, few_shot_examples: list = None):
    prompt = f"""
You are an expert in Natural Language Processing (NLP) specializing in sentiment analysis of political discourse.

Your task is to analyze the **Emotional Valence** of the text regarding the Gaza conflict.
Assign a continuous **Sentiment Score** from **-1.0** (Very Negative) to **1.0** (Very Positive).

---
**SCORING RUBRIC (GUIDELINES):**

* **-1.0 to -0.7 (Very Negative):** Extreme hostility, hate speech, violent language, insults, or deep despair/trauma.
* **-0.6 to -0.1 (Negative):** Criticism, cynicism, frustration, sadness, sarcasm, or disagreement.
* **0.0 (Neutral):** Purely factual statements, objective questions, or balanced observations without emotional loading.
* **0.1 to 0.6 (Positive):** Empathy, support, hope, agreement, or mild praise.
* **0.7 to 1.0 (Very Positive):** Strong praise, celebration, deep gratitude, enthusiasm, or relief.

**CRITICAL DISTINCTION:**
Do NOT confuse "Political Stance" with "Sentiment".
- A user can be angry (Negative Sentiment) while supporting a "Good Cause".
- A user can be hopeful (Positive Sentiment) about a controversial solution.
- Focus ONLY on the **tone and emotion** of the language used, not the validity of their opinion.

---
**EXPERT KNOWLEDGE: REFERENCE SAMPLES (Ground Truth)**
{few_shot_examples}

---
**OUTPUT FORMAT:**
Return a single JSON object.
Example: {{"sentiment_score": -0.45}}

**TEXT TO CLASSIFY:**
{content}
"""
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a sentiment analysis expert. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in Sentiment: {e}")
        return json.dumps({"sentiment_score": None})

#################################################################################################

# ==============================================================================
# Helper additional functions
# ==============================================================================

def export_labeling_samples_to_json(df, file_path, data_columns_to_show, features_to_label):
    export_list = []
    for row in df.iter_rows(named=True):
        item = {k: row[k] for k in data_columns_to_show if k in row}
        item.update({k: None for k in features_to_label})
        export_list.append(item)        
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(export_list, f, indent=4, ensure_ascii=False)
    logging.info(f"💾 File saved: {file_path}")

#==============================================================================

def load_labeled_sample(file_path):
    """Loads labeled JSON asDataFrame and filters nulls in feature_name column."""
    if not os.path.exists(file_path):
        logging.error(f"❌ File not found: {file_path}. Run script 03a/04a first.")
        exit()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not data: return pl.DataFrame([])
        df = pl.DataFrame(data)
        return df
    except Exception as e:
        logging.error(f"❌ Error loading {file_path}: {e}")
        exit()

#==============================================================================

def process_labeled_sample_for_llm(df, feature_name):
    """Converts a labeled DataFrame to formatted list to be ingested in LLMs."""
    formatted_list = []
    for row in df.iter_rows(named=True):
        formatted_list.append({
            "text_content": row['text_content'],
            feature_name: row[feature_name]
        })
    return formatted_list

#==============================================================================

def adjacent_accuracy(y_true, y_pred, adjacent_tol=1):
    """Calculates adjacent accuracy for ordinal scales."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    diff = np.abs(y_true_arr - y_pred_arr)
    adjacent_acc = np.mean(diff <= adjacent_tol)
    return adjacent_acc

#==============================================================================
# NORMALIZACIÓN
#==============================================================================
def normalize_str_categories(values):
    return [str(l).strip().lower() if l is not None else "unknown" for l in values]

#################################################################################################

#==============================================================================
# VALIDATION RUNNER (SEQUENTIAL ASYNC)
#==============================================================================

async def run_validation_for_feature(feature_name, feature_config, df_train, df_val, validation_results_dir,
                                     client, model_name, temperature): 

    if not feature_config:
        logging.error(f"❌ Configuration not found for {feature_name}")
        return
    
    logging.info(f"\n🔵 VALIDATING FEATURE: {feature_name.upper()}")
    
    if len(df_train) == 0 or len(df_val) == 0:
        logging.error("❌ Error: Missing labeled data. Check 'data/labeled_samples' folder.")
        return

    # 1. Clean nulls
    df_train = df_train.filter(pl.col(feature_name).is_not_null())
    df_val = df_val.filter(pl.col(feature_name).is_not_null())

    logging.info(f"📂 Data Loaded -> Train (Few-Shot): {len(df_train)} | Val (Test): {len(df_val)}")

    # 2. Prepare Few-Shot Examples
    few_shot_examples = process_labeled_sample_for_llm(df_train, feature_name)

    # 3. Inference (Loop secuencial con await)
    y_true = []
    y_pred = []
    
    feature_type = feature_config['type']
    validation_threshold = feature_config['validation_threshold']

    validation_results = {}
    validation_results['feature_name'] = str(feature_name)
    validation_results['feature_type'] = str(feature_type)

    logging.info(f"⏳ Running predictions on {len(df_val)} records (Async Sequential)...")

    for i, row in enumerate(df_val.iter_rows(named=True)):
        text_input = row['text_content']
        true_value = row[feature_name]
        
        try:
            # CALL TO LLM (AWAIT)
            llm_response = await feature_config['func'](
                client=client, 
                content=text_input, 
                few_shot_examples=few_shot_examples, 
                model_name=model_name,
                temperature=temperature
            )
            
            response_json = json.loads(llm_response)
            predicted_value = response_json.get(feature_name)
            
            # Safety Casting
            if feature_type == 'ordinal':
                predicted_value = int(predicted_value) if predicted_value is not None else -1
                true_value = int(true_value)
            
            elif feature_type == 'continuous':
                predicted_value = float(predicted_value) if predicted_value is not None else 0.0
                true_value = float(true_value)

            else:
                predicted_value = str(predicted_value) if predicted_value is not None else "ERROR"
                true_value = str(true_value)

        except Exception as e:
            logging.warning(f"⚠️ Error in record {i}: {e}")
            if feature_type == 'ordinal': predicted_value = -1
            elif feature_type == 'continuous': predicted_value = 0.0
            else: predicted_value = "ERROR"

        y_true.append(true_value)
        y_pred.append(predicted_value)
        
        if (i+1) % 10 == 0: print(f"   Processed {i+1}/{len(df_val)}...")

    # 4. Metrics & Reporting
    logging.info(f"📊 METRICS logging: {feature_name}")
    
    if feature_type == 'ordinal':
        validation_results['validation_score_type'] = 'accuracy'
        if feature_name != 'content_relevance_score':
            score_value = adjacent_accuracy(y_true, y_pred) 
            logging.info(f"   🎯 Adjacent Accuracy:  {score_value:.2%} (Target: >= {validation_threshold:.0%})")
        else:
            cutoff = feature_config['cutoff']
            bin_true = [1 if x >= cutoff else 0 for x in y_true]
            bin_pred = [1 if x >= cutoff else 0 for x in y_pred]
            score_value = accuracy_score(bin_true, bin_pred)
            logging.info(f"   ⚖️ Binary Filter Acc:  {score_value:.2%} (Target: >= {validation_threshold:.0%})")
        validation_passed = score_value >= validation_threshold

    elif feature_type == 'categorical':
        validation_results['validation_score_type'] = 'accuracy'
        y_true = normalize_str_categories(y_true)
        y_pred = normalize_str_categories(y_pred)
        score_value = accuracy_score(y_true, y_pred)
        logging.info(f"   🎯 Exact Accuracy:     {score_value:.2%} (Target: >= {validation_threshold:.0%})")
        validation_passed = score_value >= validation_threshold

    elif feature_type == 'continuous':
        validation_results['validation_score_type'] = 'error'
        score_value = mean_absolute_error(y_true, y_pred)
        logging.info(f"   📉 MAE: {score_value:.4f} (Target: <= {validation_threshold})")        
        validation_passed = score_value <= validation_threshold
        
    validation_results['validation_score'] = float(score_value)
    validation_results['validation_threshold'] = float(validation_threshold)
    validation_results['validation_passed'] = bool(validation_passed)

    logging.info("   ✅ SUCCESS: validation passed.") if validation_passed else logging.info("   🛑 FAILURE: validation not passed.")
   
    os.makedirs(validation_results_dir, exist_ok=True)
    validation_results_filename = f"validation_results_{feature_name}.json"
    validation_results_path = os.path.join(validation_results_dir, validation_results_filename)
    with open(validation_results_path, "w", encoding="utf-8") as f:
        json.dump(validation_results, f, ensure_ascii=False, indent=4)

#################################################################################################

#==============================================================================
# GENERATION RUNNER (PARALLEL ASYNC)
#==============================================================================

async def process_single_row(sem, row, client, feature_name, feature_config, few_shot_examples, model_name, temperature):
    """Worker para procesar una fila individual con semáforo"""
    async with sem:
        comment_id = row['comment_id']
        text_input = row['text_content']
        feature_type = feature_config['type']
        
        try:
            llm_response = await feature_config['func'](
                client=client, 
                content=text_input, 
                few_shot_examples=few_shot_examples,
                model_name=model_name,
                temperature=temperature
            )
            response_json = json.loads(llm_response)
            predicted_value = response_json.get(feature_name)
            
            if feature_type == 'ordinal':
                predicted_value = int(predicted_value) if predicted_value is not None else -1
            elif feature_type == 'continuous':
                predicted_value = float(predicted_value) if predicted_value is not None else 0.0
            else:
                predicted_value = str(predicted_value) if predicted_value is not None else "ERROR"

        except Exception as e:
            # logging.warning(f"⚠️ Error in comment {comment_id}: {e}") # Descomentar si se quiere verbose
            if feature_type == 'ordinal': predicted_value = -1
            elif feature_type == 'continuous': predicted_value = 0.0
            else: predicted_value = "ERROR"

        return {
            "comment_id": comment_id,
            feature_name: predicted_value
        }
    
##############################################################

async def run_generation_for_feature(feature_name, feature_file_path, feature_config, df, df_train, 
                                    batch_save_size, max_concurrent_request, pilot_mode, pilot_size, pilot_seed, 
                                    client, model_name, temperature): 

    mode_msg = f"🧪 PILOT MODE (Max {pilot_size} records)" if pilot_mode else "🚀 PRODUCTION MODE (Full Data)"
    logging.info(f"STARTING GENERATION of {feature_name.upper()}")
    logging.info(f"MODE: {mode_msg}")

    few_shot_examples = process_labeled_sample_for_llm(df_train, feature_name)

    # 1. PREPARE DATA
    processed_ids = set()
    if os.path.exists(feature_file_path):
        try:
            df_existing = pl.read_parquet(feature_file_path)
            processed_ids = set(df_existing['comment_id'].to_list())
            logging.info(f"🔄 Resume: Found {len(processed_ids)} processed records.")
        except Exception:
            pass

    df_to_process = df.filter(~ pl.col('comment_id').is_in(processed_ids))
    
    if pilot_mode:
        if len(df_to_process) > pilot_size:
            df_to_process = df_to_process.sample(n=pilot_size, seed=pilot_seed)
    
    records = df_to_process.to_dicts() 
    total_records = len(records)
    
    if total_records == 0:
        logging.info("✅ No new records to process. Exiting.")
        return

    logging.info(f"⏳ Processing {total_records} records with AsyncIO...")

    # 2. CONCURRENCY CONTROL
    sem = asyncio.Semaphore(max_concurrent_request)

    # 3. BATCH PROCESSING LOOP
    for i in range(0, total_records, batch_save_size):
        chunk = records[i : i + batch_save_size]
        
        tasks = [
            process_single_row(sem, row, client, feature_name, feature_config, few_shot_examples, model_name, temperature)
            for row in chunk
        ]
        
        logging.info(f"🚀 Launching batch {i} - {min(i+batch_save_size, total_records)}...")
        
        # Ejecutar tareas con barra de progreso
        results = await tqdm_asyncio.gather(*tasks)
        
        # 4. SAVE BATCH
        if results:
            df_new_chunk = pl.DataFrame(results)
            if os.path.exists(feature_file_path):
                try:
                    df_current = pl.read_parquet(feature_file_path)
                    pl.concat([df_current, df_new_chunk]).write_parquet(feature_file_path)
                except Exception as e:
                    logging.error(f"❌ Error saving batch: {e}")
            else:
                df_new_chunk.write_parquet(feature_file_path)

    logging.info("✅ Async Generation Process Completed.")


#################################################################################################

async def fetch_embeddings_for_batch(client, texts_batch, model_name):
    """Obtiene embeddings para un lote de textos en una sola llamada."""
    try:
        # Reemplazar saltos de línea es recomendación oficial de OpenAI para embeddings
        clean_texts = [t.replace("\n", " ") for t in texts_batch]
        
        resp = await client.embeddings.create(
            input=clean_texts,
            model=model_name
        )
        # La respuesta viene ordenada, extraemos los vectores
        return [data.embedding for data in resp.data]
    except Exception as e:
        logging.error(f"❌ Error in embedding batch: {e}")
        # Retorna lista de Nones del tamaño del batch para no romper índices
        return [None] * len(texts_batch)

#################################################################################################

async def process_embeddings_for_batch(sem, batch, client, model_name):
   
    async with sem:
        texts_batch = [r['text_content'] for r in batch]
        ids = [r['comment_id'] for r in batch]
        vectors = await fetch_embeddings_for_batch(client, texts_batch, model_name)
        
        batch_results = []
        for cid, vec in zip(ids, vectors):
            if vec is not None:
                batch_results.append({"comment_id": cid, "raw_embedding": vec})
        return batch_results

#################################################################################################

async def run_embedding_generation(raw_embeddings_path, df, batch_size, max_concurrent_request, client, model_name):

    # Verificamos si ya existen embeddings crudos guardados para ahorrar dinero
    if os.path.exists(raw_embeddings_path):

        df_raw = pl.read_parquet(raw_embeddings_path)
        logging.info(f"🔄 Found existing RAW embeddings file for {len(df_raw)} records. Loading...")
        
        # Identificar qué falta
        existing_ids = set(df_raw['comment_id'].to_list())
        df_to_process = df.filter(~pl.col('comment_id').is_in(existing_ids))

    else:
        
        df_raw = pl.DataFrame(schema={'comment_id': pl.Utf8, 'raw_embedding': pl.List(pl.Float64)})
        df_to_process = df

    # Si hay nuevos datos, procesamos
    if len(df_to_process) > 0:
        
        logging.info(f"⚡ Generating embeddings for {len(df_to_process)} records...")
        
        # Convertir a listas para iterar
        records = df_to_process.select(['comment_id', 'text_content']).to_dicts()
        
        # Preparar lotes
        batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
        
        new_results = []
        
        # CONCURRENCY CONTROL (Embeddings tiene rate limits altos, 50 es seguro)
        sem = asyncio.Semaphore(max_concurrent_request)

        # Ejecutar peticiones
        tasks = [
            process_embeddings_for_batch(sem=sem, batch=b, client=client, model_name=model_name) 
            for b in batches
        ]
        results_nested = await tqdm_asyncio.gather(*tasks)
        
        # Aplanar lista de listas
        for batch_res in results_nested:
            new_results.extend(batch_res)

        # Guardar Raw Embeddings
        if new_results:

            df_new = pl.DataFrame(new_results)
            
            # Unir con lo existente
            df_raw = pl.concat([df_raw, df_new], how="vertical")
            
            # Guardar en disco (Checkpoint)
            df_raw.write_parquet(raw_embeddings_path)
            logging.info(f"💾 Saved raw embeddings checkpoint to: {raw_embeddings_path}")
   
    else:
        logging.info("✅ All records already have embeddings.")

#################################################################################################

def run_reduce_embedding_dimension(df_raw_embeddings, n_pca_components, pca_embeddings_path): 

    # ==========================================================================
    #  REDUCCIÓN DE DIMENSIONALIDAD (PCA)
    # ==========================================================================

    # Convertir columna de listas polars a matriz numpy
    # (Necesitamos todos los datos para ajustar el PCA correctamente)
    embeddings_matrix = np.array(df_raw_embeddings['raw_embedding'].to_list())
    
    logging.info(f"📉 Running PCA to reduce {embeddings_matrix.shape[0]} dims -> {n_pca_components} dims...")

    # Obtenemos filas (muestras) y columnas (dimensiones originales)
    n_samples, n_features = embeddings_matrix.shape

    # PCA requiere que n_components sea menor o igual al MÍNIMO de filas o columnas
    min_shape = min(n_samples, n_features)
    if n_pca_components > min_shape:
        logging.error(
            f"❌ Cannot run PCA with n_pca_components={n_pca_components} since  n_pca_components > min(n_samples, n_features)={min_shape}. "
        )

    # Ajustar PCA
    pca = PCA(n_components=n_pca_components)
    reduced_embeddings_matrix = pca.fit_transform(embeddings_matrix)
    
    explained_variance = np.sum(pca.explained_variance_ratio_)
    logging.info(f"📊 PCA Completed. Total Explained Variance: {explained_variance:.2%}")

    # ==========================================================================
    # GUARDAR RESULTADO FINAL
    # ==========================================================================

    # Crear diccionario para el DataFrame
    pca_data = {
        "comment_id": df_raw_embeddings['comment_id']
    }
    
    # Añadir columnas dinámicas: embedding_pca_01, embedding_pca_02...
    for i in range(n_pca_components):
        col_name = f"embedding_pca_{i+1:02d}" # ej: embedding_pca_01
        pca_data[col_name] = reduced_embeddings_matrix[:, i]

    df_embeddings_pca = pl.DataFrame(pca_data)

    # Guardar
    df_embeddings_pca.write_parquet(pca_embeddings_path)
    logging.info(f"✅ Embeddings dimension reduced.")

#################################################################################################