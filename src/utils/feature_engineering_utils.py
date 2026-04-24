#################################################################################################

# Imports 

import os
import time
import json
import logging
import asyncio
import polars as pl
import numpy as np
from openai import AsyncOpenAI
from sklearn.metrics import accuracy_score, mean_absolute_error
from tqdm.asyncio import tqdm_asyncio 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt  

#################################################################################################

# Logging Configuration

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#################################################################################################

# ==============================================================================
# 1. CONTENT RELEVANCE SCORE (Filtrado) - ASYNC
# ==============================================================================

async def content_relevance_score(client: AsyncOpenAI, model_name: str = "gpt-4o-mini", temperature: float = 0.0, 
                                  content: str = None, return_prompt: bool = False):
    """
    Calcula la relevancia temática usando ejemplos Few-Shot dinámicos (Versión Async).
    """
    prompt = f"""
You are an expert content rating specialist for an academic study on public opinion regarding the Gaza conflict on Reddit.

Your task is to assign a numerical **Relevance Score** from **0 (Not Related)** to **5 (Directly Related)** to the provided Reddit comment.

---
**CRITICAL CONTEXT: THE "DISCARD" THRESHOLD (< 3)**
Keep in mind that any comment receiving a score of 0, 1, or 2 **will be completely discarded from the final analysis**. You must use these lower scores confidently to filter out any comment that does not provide useful data about the public opinion on the Gaza conflict (e.g., purely domestic US politics, generic noise, meta-Reddit discussions).

---
**TOPICAL RELEVANCE SCALE (0-5):**
* **5 - Directly Related (Core Conflict):** Explicit discussion of the core conflict, everyday life in Gaza, humanitarian aid, or the main actors (Israel, Hamas, IDF, Gaza) on the ground. (e.g., civilian life, death tolls, direct war events).
* **4 - Clearly Related (Media, Opinion, Relations):** Discussions focused on media coverage of the conflict, public opinion/protests regarding the conflict, or international/military relations with Israel. Includes brief reactions/insults directed specifically at these topics.
* **3 - Marginal Context:** Historical, political explanations related to the geographical area. Broad context.
* **2 - Accidental/Trivial / Meta-Reddit:** Meta-commentary about how Reddit works (e.g., listing subreddits), keywords used in non-political context, or trivial noise.
* **1 - Off-Topic Tangential:** Broad ideological discussions not directly tied to the current conflict (e.g., general media antisemitism, WWII comparisons) or personal attacks unrelated to the topic.
* **0 - Discard / Internal Politics / Generic:** Focuses on US Internal Affairs/politicians, generic/unclear short comments ("badass"), or completely unrelated content.

---
**OUTPUT FORMAT:**
Return a single JSON object. 
Keys:
- "reasoning_content_relevance_score": A concise explanation (1-2 sentences). Step 1: Identify the primary subject (e.g., US Elections, Ground War, Media). Step 2: Apply the scale or veto rules to justify the score.
- "content_relevance_score": The integer score (0-5).

Example: 
{{
  "reasoning_content_relevance_score": "The comment discusses the release of hostages and the implications of genocide accusations, which directly address the core events and legal battles of the Gaza conflict.",
  "content_relevance_score": 5
}}

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
        text_response = response.choices[0].message.content 
        return text_response if not return_prompt else (text_response, prompt)
    except Exception as e:
        logging.error(f"Error in OpenAI API call (Relevance): {e}")
        return json.dumps({"content_relevance_score": None})

#################################################################################################

# ==============================================================================
# 2. POLITICAL STANCE SCORE - ASYNC
# ==============================================================================

async def political_stance_score(client: AsyncOpenAI,  model_name: str = "gpt-4o-mini", temperature: float = 0.0, 
                                 content: str = None, return_prompt: bool = False):
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
**OUTPUT FORMAT:**
Return a single JSON object. 
Keys:
- "reasoning_political_stance_score": A concise explanation (1-2 sentences) linking the text to specific scale criteria.
- "political_stance_score": The integer score (1-5).
Example: 
{{
    "reasoning_political_stance_score": "The comment justifies the creation of Israel through historical conquest rights ('technically their land') and frames the partition as 'generous', while assigning blame for the initial conflict solely to Arabs. This is 'Strongly Pro-Israel'.",
    "political_stance_score": 5
}}

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
        text_response = response.choices[0].message.content 
        return text_response if not return_prompt else (text_response, prompt)
    except Exception as e:
        logging.error(f"Error in OpenAI API call (Stance): {e}")
        return json.dumps({"political_stance": None})

#################################################################################################

# ==============================================================================
# 3. DISCOURSE TONE - ASYNC
# ==============================================================================

async def discourse_tone_score(client: AsyncOpenAI,  model_name: str = "gpt-4o-mini", temperature: float = 0.0, 
                               content: str = None, return_prompt: bool = False):
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
**OUTPUT FORMAT:**
Return a single JSON object. 
Keys:
- "reasoning_discourse_tone_score": A concise explanation (1-2 sentences) linking the text to specific scale criteria.
- "discourse_tone_score": The exact category name from the list above.
Example: 
{{
    "reasoning_discourse_tone_score": "The user attempts an objective historical explanation, citing empires and wars to justify a position. It avoids overt emotional language.",
    "discourse_tone_score": "Analytical"
}}

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
        text_response = response.choices[0].message.content 
        return text_response if not return_prompt else (text_response, prompt)
    except Exception as e:
        logging.error(f"Error in Tone: {e}")
        return json.dumps({"discourse_tone": None})

#################################################################################################

# ==============================================================================
# 4. DOMINANT FRAME - ASYNC
# ==============================================================================

async def dominant_frame_score(client: AsyncOpenAI, model_name: str = "gpt-4o-mini", temperature: float = 0.0, 
                               content: str = None, return_prompt: bool = False):
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
**OUTPUT FORMAT:**
Return a single JSON object. 
Keys:
- "reasoning_dominant_frame_score": A concise explanation (1-2 sentences) linking the text to specific scale criteria.
- "dominant_frame_score": The exact category name from the list above.
Example: 
{{
    "reasoning_dominant_frame_score": "The entire argument relies on 1948/pre-1948 history (Ottoman Empire, British Mandate) to explain the present legitimacy.",
    "dominant_frame_score": "Historical/Religious"
}}

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
        text_response = response.choices[0].message.content 
        return text_response if not return_prompt else (text_response, prompt)
    except Exception as e:
        logging.error(f"Error in Frame: {e}")
        return json.dumps({"dominant_frame": None})

#################################################################################################

# ==============================================================================
# 5. ARGUMENT QUALITY SCORE - ASYNC
# ==============================================================================

async def argument_quality_score(client: AsyncOpenAI, model_name: str = "gpt-4o-mini", temperature: float = 0.0, 
                                 content: str = None, return_prompt: bool = False):
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
**OUTPUT FORMAT:**
Return a single JSON object. 
Keys:
- "reasoning_argument_quality_score": A concise explanation (1-2 sentences) linking the text to specific scale criteria.
- "argument_quality_score": The integer score (0-5).
Example: 
{{
    "reasoning_argument_quality_score": "A justified opinion that cites specific factors (water, power, borders) as obstacles to peace. Concise but logical.",
    "argument_quality_score": 3
}}

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
        text_response = response.choices[0].message.content 
        return text_response if not return_prompt else (text_response, prompt)
    except Exception as e:
        logging.error(f"Error in Quality: {e}")
        return json.dumps({"argument_quality_score": None})

#################################################################################################

# ==============================================================================
# 5. SENTIMENT SCORE - ASYNC
# ==============================================================================

async def sentiment_score(client: AsyncOpenAI, model_name: str = "gpt-4o-mini", temperature: float = 0.0, 
                          content: str = None, return_prompt: bool = False):
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
**OUTPUT FORMAT:**
Return a single JSON object. 
Keys:
- "reasoning_sentiment_score": A concise explanation (1-2 sentences) linking the text to specific scale criteria.
- "sentiment_score": The integer score (0-5).
Example: 
{{
    "reasoning_sentiment_score": "The language expresses deep pessimism and warning ('razed to the ground', 'breeds terrorism'). The tone is fearful and frustrated, falling into the negative to very negative range.",
    "sentiment_score": -0.7
}}
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
        text_response = response.choices[0].message.content 
        return text_response if not return_prompt else (text_response, prompt)
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
        item.update({f"reasoning_{k}": None for k in features_to_label})
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
            feature_name: row[feature_name],
            "reasoning": row[f"reasoning_{feature_name}"]
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
# LABELLING SAMPLES
#==============================================================================

def run_labeling_samples(df, data_columns_to_include, features_to_label, 
                         val_n, sample_seed, manual_val_ids, val_sample_path): 

    logging.info("⚙️ Starting generation of VALIDATION samples (Manual + Random)...")

    # 1. EXTRACT MANUAL SAMPLES
    df_manual_val = df.filter(pl.col('comment_id').is_in(manual_val_ids))
    
    # Check if we found all requested IDs
    if len(df_manual_val) < len(manual_val_ids):
        found = df_manual_val['comment_id'].to_list()
        missing = set(manual_val_ids) - set(found)
        logging.warning(f"⚠️ Some MANUAL VAL IDs were not found in dataset: {missing}")

    logging.info(f"🔧 Manual Validation Samples Extracted -> {len(df_manual_val)}")

    # 2. PREPARE RANDOM POOL (Excluding Manual IDs to avoid duplicates/leakage)
    df_pool = df.filter(~pl.col('comment_id').is_in(manual_val_ids))
    
    # 3. CALCULATE QUOTAS
    manual_val_n = len(df_manual_val)
    random_val_n = max(0, val_n - manual_val_n)
    logging.info(f"🎲 Random Validation Samples Needed -> {random_val_n}")

    # 4. SAMPLE RANDOM DATA
    try:
        df_random_val = df_pool.sample(n=random_val_n, seed=sample_seed, with_replacement=False)
    except Exception:
        logging.warning("⚠️ Pool is smaller than requested samples. Taking everything available.")
        df_random_val = df_pool

    # 5. COMBINE MANUAL + RANDOM
    df_val_final = pl.concat([df_manual_val, df_random_val])

    logging.info(f"📊 FINAL VALIDATION SET -> {len(df_val_final)} samples")

    # 6. EXPORT TO JSON
    export_labeling_samples_to_json(df_val_final, val_sample_path, data_columns_to_include, features_to_label)

    logging.info(f"✅ Process completed. Check {val_sample_path}")

#==============================================================================
# VALIDATION WORKER (ASYNC)
#==============================================================================

async def validate_single_row(sem, row, client, feature_name, feature_config, 
                             model_name, temperature):
    """
    Worker para validar una sola fila con control de concurrencia.
    Mantiene la misma estructura que el proceso de generación.
    """
    async with sem:
        comment_id = row['comment_id']
        text_input = row['text_content']
        true_value = row[feature_name]
        feature_type = feature_config['type']

        try:
            # Llamada al LLM
            llm_response = await feature_config['func'](
                client=client, 
                content=text_input, 
                model_name=model_name,
                temperature=temperature,
                return_prompt=False
            )
            
            response_json = json.loads(llm_response)
            predicted_value = response_json.get(feature_name)

            # Safety Casting (consistente con el runner de generación)
            if feature_type == 'ordinal':
                predicted_value = int(predicted_value) if predicted_value is not None else -1
                true_value = int(true_value)
            elif feature_type == 'continuous':
                predicted_value = float(predicted_value) if predicted_value is not None else 0.0
                true_value = float(true_value)
            else:
                predicted_value = str(predicted_value) if predicted_value is not None else "ERROR"
                true_value = str(true_value)

            return {
                'comment_id': comment_id,
                'true_value': true_value,
                'predicted_value': predicted_value,
                'raw_response': llm_response
            }

        except Exception as e:
            logging.warning(f"⚠️ Error in record {comment_id}: {e}")
            fallback_val = -1 if feature_type == 'ordinal' else (0.0 if feature_type == 'continuous' else "ERROR")
            return {
                'comment_id': comment_id,
                'true_value': true_value,
                'predicted_value': fallback_val,
                'error': str(e)
            }

#==============================================================================
# VALIDATION RUNNER (PARALLEL BATCH ASYNC)
#==============================================================================

async def run_validation_for_feature(feature_name, feature_config, df_val, 
                                     validation_results_dir, client, model_name, temperature,
                                     n_validation_iterations, global_validation_threshold,
                                     max_concurrent_request, batch_size): 
    """
    Orquestador de validación que imita el comportamiento de generación:
    Procesa en batches y usa un semáforo para control de concurrencia.
    """
    if not feature_config:
        logging.error(f"❌ Configuration not found for {feature_name}")
        return
    
    os.makedirs(validation_results_dir, exist_ok=True)
    validation_results_path = os.path.join(validation_results_dir, f"validation_results_{feature_name}.json")

    if os.path.exists(validation_results_path):
        logging.warning(f"⛔ FEATURE {feature_name.upper()} ALREADY VALIDATED")
        return
    
    logging.info(f"\n🔵 VALIDATING FEATURE: {feature_name.upper()}")
    
    if len(df_val) == 0:
        logging.error("❌ Error: Missing labeled data.")
        return

    # 1. Preparación de datos
    df_val = df_val.filter(pl.col(feature_name).is_not_null())
    
    records = df_val.to_dicts()
    total_records = len(records)
    feature_type = feature_config['type']
    validation_threshold = feature_config['validation_threshold']

    validation_results = {
        'feature_name': str(feature_name),
        'feature_type': str(feature_type),
        'individual_validation': {
            'score_type': None,
            'validation_threshold': float(validation_threshold), 
            'score_value': [],
            'validation_passed': [],
        },
        'global_validation': {
            'validation_threshold': float(global_validation_threshold),
            'prob_validation_passed': None,
            'validation_passed': None
        },

        'llm_metadata': {'model_name': model_name, 'temperature': temperature, 'iterations_predicted_values': []},
    }

    # Control de concurrencia (Semáforo compartido entre batches de una misma iteración)
    sem = asyncio.Semaphore(max_concurrent_request)

    # 2. Bucle de Iteraciones
    iterations_predicted_values = []
    for iter_idx in range(n_validation_iterations): 
        logging.info(f"⏳ ITERATION {iter_idx}: Processing {total_records} records in batches of {batch_size}...")
        time.sleep(3)
        
        all_iter_results = []

        # 3. Procesamiento por Batches (Imitando Generación)
        for i in range(0, total_records, batch_size):
            chunk = records[i : i + batch_size]
            
            tasks = [
                validate_single_row(sem, row, client, feature_name, feature_config, 
                                    model_name, temperature)
                for row in chunk
            ]
            
            # Ejecución del batch
            batch_results = await tqdm_asyncio.gather(*tasks, desc=f"Iter {iter_idx} | Batch {i//batch_size}")
            all_iter_results.extend(batch_results)

        # 4. Cálculo de métricas tras completar la iteración
        y_true = [r['true_value'] for r in all_iter_results]
        y_pred = [r['predicted_value'] for r in all_iter_results]
        
        iterations_predicted_values.append(
            [r['predicted_value'] for r in all_iter_results]
        )

        if feature_type == 'ordinal':
            validation_results['individual_validation']['score_type'] = 'accuracy'
            if feature_name != 'content_relevance_score':
                score_value = adjacent_accuracy(y_true, y_pred) 
                logging.info(f"  🎯 Iter {iter_idx} - Adj. Acc: {score_value:.2%}")
            else:
                cutoff = feature_config['cutoff']
                bin_true = [1 if x >= cutoff else 0 for x in y_true]
                bin_pred = [1 if x >= cutoff else 0 for x in y_pred]
                score_value = accuracy_score(bin_true, bin_pred)
                logging.info(f"  ⚖️ Iter {iter_idx} - Binary Acc: {score_value:.2%}")
            
            passed = score_value >= validation_threshold

        elif feature_type == 'categorical':
            validation_results['individual_validation']['score_type'] = 'accuracy'
            score_value = accuracy_score(normalize_str_categories(y_true), normalize_str_categories(y_pred))
            logging.info(f"  🎯 Iter {iter_idx} - Accuracy: {score_value:.2%}")
            passed = score_value >= validation_threshold

        elif feature_type == 'continuous':
            validation_results['individual_validation']['score_type'] = 'error'
            score_value = mean_absolute_error(y_true, y_pred)
            logging.info(f"  📉 Iter {iter_idx} - MAE: {score_value:.4f}")
            passed = score_value <= validation_threshold
        
        validation_results['individual_validation']['score_value'].append(float(score_value))
        validation_results['individual_validation']['validation_passed'].append(bool(passed))

    
    comment_ids = df_val['comment_id'].to_list()
    validation_results['llm_metadata']['iterations_predicted_values'] = {
        c: [x[c_idx] for x in iterations_predicted_values] 
        for c_idx, c in enumerate(comment_ids)
        }

    # 5. Lógica de Validación Global
    prob_passed = np.mean(validation_results['individual_validation']['validation_passed'])
    global_passed = prob_passed >= global_validation_threshold
    
    validation_results['global_validation']['prob_validation_passed'] = float(prob_passed)
    validation_results['global_validation']['validation_passed'] = bool(global_passed)

    logging.info(f"{'✅' if global_passed else '🛑'} FEATURE {feature_name.upper()} VALIDATION {'PASSED' if global_passed else 'FAILED'} ({prob_passed:.0%})")
   
    with open(validation_results_path, "w", encoding="utf-8") as f:
        json.dump(validation_results, f, ensure_ascii=False, indent=4)

#################################################################################################

#==============================================================================
# GENERATION RUNNER (PARALLEL ASYNC)
#==============================================================================

async def process_single_row(sem, row, client, feature_name, feature_config, 
                             model_name, temperature, 
                             metadata_file_path, file_lock):
    """Worker para procesar una fila individual con semáforo"""

    async with sem:
        comment_id = row['comment_id']
        text_input = row['text_content']
        feature_type = feature_config['type']
        
        start_time = time.perf_counter()

        try:
            llm_response, llm_prompt = await feature_config['func'](
                client=client, 
                content=text_input, 
                model_name=model_name,
                temperature=temperature,
                return_prompt=True
            )
            response_json = json.loads(llm_response)
            predicted_value = response_json.get(feature_name)
            reasoning = response_json.get(f"reasoning_{feature_name}")
            
            if feature_type == 'ordinal':
                predicted_value = int(predicted_value) if predicted_value is not None else -1
            elif feature_type == 'continuous':
                predicted_value = float(predicted_value) if predicted_value is not None else 0.0
            else:
                predicted_value = str(predicted_value) if predicted_value is not None else "ERROR"

        except Exception as e:
            logging.warning(f"⚠️ Error in comment {comment_id}: {e}") 
            if feature_type == 'ordinal': predicted_value = -1
            elif feature_type == 'continuous': predicted_value = 0.0
            else: predicted_value = "ERROR"

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        response = {
            "comment_id": comment_id,
            feature_name: predicted_value,
            f"reasoning_{feature_name}": reasoning
        }
        
        current_metadata = {
            'comment_id': comment_id,
            'response': llm_response,
            'time': round(elapsed_time, 4),
            'input_tokens': len(str(llm_prompt)) // 4,
            'output_tokens': len(str(llm_response)) // 4,
            'model_name': model_name,
            'temperature': temperature,
            'processing_date': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        async with file_lock:
            feature_metadata = {}
            if os.path.exists(metadata_file_path):
                try:
                    with open(metadata_file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if content:
                            feature_metadata = json.loads(content)
                except json.JSONDecodeError:
                    logging.error(f"Error decoding JSON from {metadata_file_path}, starting fresh.")
                    feature_metadata = {}
            feature_metadata[str(comment_id)] = current_metadata
            with open(metadata_file_path, "w", encoding="utf-8") as f:
                json.dump(feature_metadata, f, ensure_ascii=False, indent=4)             

        return response
    
##############################################################

async def run_generation_for_feature(feature_name, feature_file_path, feature_config, df, 
                                    batch_save_size, max_concurrent_request,  
                                    client, model_name, temperature, metadata_file_path, file_lock,
                                    pilot_mode=None, pilot_size=None, pilot_seed=None): 

    logging.info(f"▶️ STARTING GENERATION of {feature_name.upper()}")
    mode_msg = f"🧪 PILOT MODE (Max {pilot_size} records)" if pilot_mode else "🚀 PRODUCTION MODE (Full Data)"
    logging.info(f"MODE: {mode_msg}")

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
        time.sleep(3)
        chunk = records[i : i + batch_save_size]
        
        tasks = [
            process_single_row(sem, row, client, feature_name, feature_config, model_name, temperature, metadata_file_path, file_lock)
            for row in chunk
        ]
        
        logging.info(f"🚀 Launching batch {i} - {min(i+batch_save_size, total_records)}...")
        
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

async def process_embeddings_for_batch(sem, batch, client, model_name, metadata_file_path, file_lock):

    start_time = time.perf_counter()

    async with sem:
        texts_batch = [r['text_content'] for r in batch]
        ids = [r['comment_id'] for r in batch]
        vectors = await fetch_embeddings_for_batch(client, texts_batch, model_name)

    end_time = time.perf_counter()
    total_elapsed = end_time - start_time
    
    batch_results = []
    batch_metadata_updates = {}  
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    for i, (cid, vec) in enumerate(zip(ids, vectors)):
        if vec is not None:
            # Resultado principal
            batch_results.append({"comment_id": cid, "raw_embedding": vec})
            
            text_len = len(texts_batch[i])
            avg_time = total_elapsed / len(ids) if ids else 0 
            
            batch_metadata_updates[str(cid)] = {
                'comment_id': cid,
                'time': round(avg_time, 5),
                'input_tokens': text_len // 4, 
                'model_name': model_name,
                'processing_date': timestamp,
            }  

    if batch_metadata_updates:
        async with file_lock:
            feature_metadata = {}
            
            # Leer existente
            if os.path.exists(metadata_file_path):
                try:
                    with open(metadata_file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if content:
                            feature_metadata = json.loads(content)
                except json.JSONDecodeError:
                    logging.warning(f"Error decoding JSON {metadata_file_path}, resetting.")
                    feature_metadata = {}

            # Actualizar masivamente (Update dict merges keys)
            feature_metadata.update(batch_metadata_updates)

            # Escribir todo de vuelta
            with open(metadata_file_path, "w", encoding="utf-8") as f:
                json.dump(feature_metadata, f, ensure_ascii=False, indent=4)        
    
        
    return batch_results

#################################################################################################

async def run_embedding_generation(raw_embeddings_path, df, batch_size, max_concurrent_request, 
                                   client, model_name, metadata_file_path, file_lock,
                                   pilot_mode=None, pilot_size=None, pilot_seed=None): 

    # 0. SETUP DE MODOS Y LOGS
    mode_msg = f"🧪 PILOT MODE (Max {pilot_size} records)" if pilot_mode else "🚀 PRODUCTION MODE (Full Data)"
    logging.info(f"STARTING EMBEDDING GENERATION")
    logging.info(f"MODE: {mode_msg}")

    # 1. PREPARAR DATOS (RESUME LOGIC)
    # Verificamos si ya existen embeddings crudos guardados
    if os.path.exists(raw_embeddings_path):
        try:
            df_raw = pl.read_parquet(raw_embeddings_path)
            existing_ids = set(df_raw['comment_id'].to_list())
            logging.info(f"🔄 Resume: Found {len(existing_ids)} existing embeddings.")
        except Exception as e:
            logging.warning(f"⚠️ Could not read existing file: {e}. Starting fresh.")
            df_raw = pl.DataFrame(schema={'comment_id': pl.Utf8, 'raw_embedding': pl.List(pl.Float64)})
            existing_ids = set()
    else:
        df_raw = pl.DataFrame(schema={'comment_id': pl.Utf8, 'raw_embedding': pl.List(pl.Float64)})
        existing_ids = set()

    # Identificar qué falta por procesar
    df_to_process = df.filter(~pl.col('comment_id').is_in(existing_ids))

    # 2. APLICAR LÓGICA PILOT MODE
    if pilot_mode:
        if len(df_to_process) > pilot_size:
            logging.info(f"✂️ Downsampling from {len(df_to_process)} to {pilot_size} for Pilot Mode...")
            df_to_process = df_to_process.sample(n=pilot_size, seed=pilot_seed)

    # 3. PROCESAMIENTO
    if len(df_to_process) > 0:
        
        logging.info(f"⚡ Generating embeddings for {len(df_to_process)} records...")
        
        # Convertir a listas para iterar
        records = df_to_process.select(['comment_id', 'text_content']).to_dicts()
        
        # Preparar lotes
        batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
        
        new_results = []
        
        # CONCURRENCY CONTROL
        sem = asyncio.Semaphore(max_concurrent_request)

        # Ejecutar peticiones
        tasks = [
            process_embeddings_for_batch(
                sem=sem, 
                batch=b, 
                client=client, 
                model_name=model_name,
                metadata_file_path=metadata_file_path, 
                file_lock=file_lock
            ) 
            for b in batches
        ]
        
        # Esperar resultados
        results_nested = await tqdm_asyncio.gather(*tasks)
        
        # Aplanar lista de listas (batch -> items)
        for batch_res in results_nested:
            new_results.extend(batch_res)

        # 4. GUARDADO (MERGE & SAVE)
        if new_results:
            df_new = pl.DataFrame(new_results)
            
            # Unir con lo existente (Vertical Concatenation)
            # Nota: Si el archivo original estaba vacío o no existía, df_raw tiene el esquema correcto.
            df_final = pl.concat([df_raw, df_new], how="vertical")
            
            # Guardar en disco
            df_final.write_parquet(raw_embeddings_path)
            logging.info(f"💾 Saved {len(df_new)} new embeddings to: {raw_embeddings_path}")
            logging.info(f"📊 Total records in file: {len(df_final)}")
    
    else:
        logging.info("✅ No new records to process (or Pilot limit reached).")

#################################################################################################

def run_reduce_embedding_dimension(df_raw_embeddings, pca_embeddings_path): 

    # Convertir columna de listas polars a matriz numpy
    embeddings_matrix = np.array(df_raw_embeddings['raw_embedding'].to_list())
    n_samples, n_features = embeddings_matrix.shape
    
    logging.info(f"📉 Analyzing PCA spectrum for {n_samples} samples with {n_features} features...")

    # ==========================================================================
    # 1. FASE DE ANÁLISIS VISUAL
    # ==========================================================================
    
    # Calculamos un máximo razonable para visualizar (ej: 50 o el total de features)
    # No hace falta calcular 1000 componentes si solo nos interesa ver dónde se aplana la curva
    max_components_viz = min(n_samples, n_features, 50) 
    
    pca_viz = PCA(n_components=max_components_viz)
    pca_viz.fit(embeddings_matrix)
    
    cumulative_variance = np.cumsum(pca_viz.explained_variance_ratio_)

    # Generar gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_components_viz + 1), cumulative_variance, marker='o', linestyle='--')
    plt.axhline(y=0.90, color='r', linestyle=':', label='90% Variance')
    plt.axhline(y=0.95, color='g', linestyle=':', label='95% Variance')
    
    plt.title('Explained Variance vs. Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.legend()
    plt.grid(True)
    
    print("\n" + "="*60)
    print(f"📊 GRAPH GENERATED. Close the plot window to continue.")
    print("="*60)
    
    #  - Esto mostrará el gráfico y bloqueará el script hasta cerrarlo
    plt.show() 

    # ==========================================================================
    # 2. SELECCIÓN DE USUARIO
    # ==========================================================================
    
    while True:
        try:
            user_input = input(f"\n👉 Enter the desired number of PCA components (1-{min(n_samples, n_features)}): ")
            n_pca_components = int(user_input)
            if 1 <= n_pca_components <= min(n_samples, n_features):
                break
            else:
                print(f"❌ Invalid range. Please enter a number between 1 and {min(n_samples, n_features)}.")
        except ValueError:
            print("❌ Invalid input. Please enter an integer.")

    # ==========================================================================
    # 3. REDUCCIÓN FINAL Y GUARDADO
    # ==========================================================================

    logging.info(f"📉 Applying Final PCA with n_components={n_pca_components}...")
    
    pca_final = PCA(n_components=n_pca_components)
    reduced_embeddings_matrix = pca_final.fit_transform(embeddings_matrix)
    
    final_variance = np.sum(pca_final.explained_variance_ratio_)
    logging.info(f"📊 PCA Completed. Total Explained Variance: {final_variance:.2%}")

    # Crear diccionario para el DataFrame
    pca_data = {
        "comment_id": df_raw_embeddings['comment_id']
    }
    
    for i in range(n_pca_components):
        col_name = f"embedding_pca_{i+1:02d}"
        pca_data[col_name] = reduced_embeddings_matrix[:, i]

    df_embeddings_pca = pl.DataFrame(pca_data)

    df_embeddings_pca.write_parquet(pca_embeddings_path)
    logging.info(f"✅ Embeddings dimension reduced to {n_pca_components} and saved.")

#################################################################################################

def get_data_for_features_validation_analysis(df_val, val_results, feature_name):

    true_values_df = df_val[['comment_id', feature_name]]

    iter_pred_values_dict = val_results['llm_metadata']['iterations_predicted_values']
    iter_pred_values_df_wide = pl.DataFrame(iter_pred_values_dict)
    comment_ids = list(iter_pred_values_dict.keys())
    predicted_values = list(iter_pred_values_dict.values())
    iter_pred_values_df_long = pl.DataFrame({'comment_id': comment_ids, 'predicted_values': predicted_values})

    true_predicted_values_df = true_values_df.join(
        iter_pred_values_df_long, on='comment_id', how='inner'
    ).with_columns(
        predicted_values_mean = pl.col('predicted_values').list.mean(),
        predicted_values_std = pl.col('predicted_values').list.std(),
        predicted_values_q25 = pl.col('predicted_values').list.eval(pl.element().quantile(0.25)).list.first(),
        predicted_values_q75 = pl.col('predicted_values').list.eval(pl.element().quantile(0.75)).list.first(),
        predicted_values_range = 5 - 0 # Theoretical range 
        # predicted_values_range = pl.col('predicted_values').list.max() - pl.col('predicted_values').list.min() # Observed range
    ).with_columns(
        predicted_values_cv_std = (pl.col('predicted_values_std') / pl.col('predicted_values_mean').abs()).fill_nan(None),
        predicted_values_cv_quantiles = ((pl.col('predicted_values_q75') - pl.col('predicted_values_q25')) / (pl.col('predicted_values_q75') + pl.col('predicted_values_q25'))).fill_nan(None),
        predicted_values_cv_range = (pl.col('predicted_values_std') / pl.col('predicted_values_range')).fill_nan(None)
    ).sort(
        by='predicted_values_mean'
    )

    df_plot = iter_pred_values_df_wide.unpivot(
        variable_name="comment_id", 
        value_name="prediction"
    ).sort(
        by='prediction'
    ).join(true_values_df, on='comment_id', how='inner'
    ).join(true_predicted_values_df[['comment_id', 'predicted_values_mean', 'predicted_values_std']], on='comment_id', how='inner'
    )

    return true_predicted_values_df, df_plot

#################################################################################################