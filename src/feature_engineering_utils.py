import os
import json
import logger
import datetime
import polars as pl
from openai import OpenAI
from sklearn.metrics import accuracy_score, mean_absolute_error

# Configurar logger
logger.basicConfig(level=logger.INFO, format='%(levelname)s: %(message)s')

# ==============================================================================
# 1. CONTENT RELEVANCE SCORE (Filtrado)
# ==============================================================================

def content_relevance_score(client: OpenAI, content: str, few_shot_examples: list = None):
    """
    Calcula la relevancia temática usando ejemplos Few-Shot dinámicos.
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
        response = client.chat.completions.create(
            model="gpt-4o-mini", # O el modelo que estés usando
            messages=[
                {"role": "system", "content": "You are a helpful classification assistant. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
            temperature=0.0 # Temperatura 0 para máxima consistencia y reproducibilidad
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in OpenAI API call (Relevance): {e}")
        return json.dumps({"content_relevance_score": None})


# ==============================================================================
# 2. POLITICAL STANCE SCORE
# ==============================================================================

def political_stance_score(client: OpenAI, content: str, few_shot_examples: list = None):
    """
    Calcula la postura política usando ejemplos Few-Shot dinámicos.
    """

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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a political analyst. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in OpenAI API call (Stance): {e}")
        return json.dumps({"political_stance": None})
    
# ==============================================================================
# 3. DISCOURSE TONE (Nuevo)
# ==============================================================================

def discourse_tone_score(client: OpenAI, content: str, few_shot_examples: list = None):
    """
    Identifica el tono dominante del discurso (Categórica Nominal).
    """

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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a linguist. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in Tone: {e}")
        return json.dumps({"discourse_tone": None})


# ==============================================================================
# 4. DOMINANT FRAME (Nuevo)
# ==============================================================================

def dominant_frame_score(client: OpenAI, content: str, few_shot_examples: list = None):
    """
    Identifica el marco retórico o temático principal (Categórica Nominal).
    """

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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a media analyst. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in Frame: {e}")
        return json.dumps({"dominant_frame": None})


# ==============================================================================
# 5. ARGUMENT QUALITY SCORE (Nuevo)
# ==============================================================================

def argument_quality_score(client: OpenAI, content: str, few_shot_examples: list = None):
    """
    Evalúa la calidad y sofisticación del argumento (Ordinal 0-5).
    """

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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a researcher. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in Quality: {e}")
        return json.dumps({"argument_quality_score": None})

# ==============================================================================
# 5. SENTIMENT SCORE
# ==============================================================================

def sentiment_score(client: OpenAI, content: str, few_shot_examples: list = None):

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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis expert. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in Quality: {e}")
        return json.dumps({"argument_quality_score": None})
    
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
    logger.info(f"💾 File saved: {file_path}")

#==============================================================================

def load_labeled_sample(file_path):
    """Loads labeled JSON asDataFrame and filters nulls in feature_name column."""

    if not os.path.exists(file_path):
        logger.error(f"❌ File not found: {file_path}. Run script 03a/04a first.")
        exit()

    try:
        # Read json as dict
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data: return pl.DataFrame([])
        
        # Read dict as df
        df = pl.DataFrame(data)
        
        return df
    
    except Exception as e:
        logger.error(f"❌ Error loading {file_path}: {e}")
        exit()

#==============================================================================

def process_labeled_sample_for_llm(df, feature_name):
    """Converts a labeled DataFrame to formatted list to be ingested in LLMs."""

    formatted_list = []
    for row in df.iter_rows(named=True):
        formatted_list.append({
            "text_content": row['text_content'], # Key used by your utils
            feature_name: row[feature_name]
        })

    return formatted_list

#==============================================================================

def adjacent_accuracy(y_true, y_pred, adjacent_tol=1):
    """Calculates adjacent accuracy for ordinal scales."""
    
    # Adjacent Tolerance (+/- adjacent_tol)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    diff = np.abs(y_true_arr - y_pred_arr)
    adjacent_acc = np.mean(diff <= adjacent_tol)
    return adjacent_acc

#==============================================================================

def run_labeling_samples(df, data_columns_to_include, features_to_label, 
                         sample_n, sample_seed, val_sample_ratio, manual_train_ids, manual_val_ids,
                         train_sample_path, val_sample_path): 

    logger.info("⚙️ Starting generation of samples (Manual + Random)...")

    # 2. EXTRACT MANUAL SAMPLES
    # Identify manual rows
    df_manual_train = df.filter(pl.col('comment_id').is_in(manual_train_ids))
    df_manual_val = df.filter(pl.col('comment_id').is_in(manual_val_ids))
    
    # Check if we found all requested IDs
    if len(df_manual_train) < len(manual_train_ids):
        found = df_manual_train['comment_id'].to_list()
        missing = set(manual_train_ids) - set(found)
        logger.warning(f"⚠️ Some MANUAL TRAIN IDs were not found in dataset: {missing}")

    if len(df_manual_val) < len(manual_val_ids):
        found = df_manual_val['comment_id'].to_list()
        missing = set(manual_val_ids) - set(found)
        logger.warning(f"⚠️ Some MANUAL VAL IDs were not found in dataset: {missing}")

    logger.info(f"🔧 Manual Samples Extracted -> Train: {len(df_manual_train)} | Val: {len(df_manual_val)}")

    # 3. PREPARE RANDOM POOL (Excluding Manual IDs to avoid duplicates/leakage)
    all_manual_ids = manual_train_ids + manual_val_ids
    df_pool = df.filter(~ pl.col('comment_id').is_in(all_manual_ids))
    
    # 4. CALCULATE QUOTAS
    val_n = int(sample_n * val_sample_ratio)
    train_n = sample_n - val_n   
    manual_val_n = len(df_manual_val)
    manual_train_n = len(df_manual_train)
    
    # Calculate how many randoms we still need
    random_train_n = max(0, train_n - manual_train_n)
    random_val_n = max(0, val_n - manual_val_n)
    random_total_n = random_train_n + random_val_n
    logger.info(f"🎲 Random Samples Needed -> Train: {random_train_n} | Val: {random_val_n}")

    # 5. SAMPLE RANDOM DATA
    try:
        df_random_selected = df_pool.sample(n=random_total_n, seed=sample_seed, with_replacement=False)
    except Exception:
        logger.warning("⚠️ Pool is smaller than requested samples. Taking everything available.")
        df_random_selected = df_pool

    # Split the random selection into train and val chunks
    df_random_train = df_random_selected[:random_train_n]
    df_random_val = df_random_selected[random_train_n:]

    # 6. COMBINE MANUAL + RANDOM
    df_train_final = pl.concat([df_manual_train, df_random_train])
    df_val_final = pl.concat([df_manual_val, df_random_val])

    logger.info(f"📊 FINAL SPLIT -> Train (Few-Shot): {len(df_train_final)} | Validation (Blind): {len(df_val_final)}")

    # 7. EXPORT TO JSON
    export_labeling_samples_to_json(df_train_final, train_sample_path, data_columns_to_include, features_to_label)
    export_labeling_samples_to_json(df_val_final, val_sample_path, data_columns_to_include, features_to_label)

    logger.info("✅ Process completed. Check 'data/labeled_samples' folder.")

#==============================================================================

def run_validation_for_feature(feature_name, feature_config, df_train, df_val, client, logger): 

    if not feature_config:
        logger.log(f"❌ Configuration not found for {feature_name}")
        return
    
    logger.log(f"\n🔵 VALIDATING FEATURE: {feature_name.upper()} ({feature_config['type']})")
    
    if len(df_train) == 0 or len(df_val) == 0:
        logger.log("❌ Error: Missing labeled data. Check 'data/labeled_samples' folder.")
        return

    # 1. Clean nulls from feature_name column in df_train and df_val 
    df_train = df_train.filter(pl.col(feature_name).is_not_null())
    df_val = df_val.filter(pl.col(feature_name).is_not_null())

    logger.log(f"📂 Data Loaded -> Train (Few-Shot): {len(df_train)} | Val (Test): {len(df_val)}")

    # 2. Prepare Few-Shot Examples
    few_shot_examples = process_labeled_sample_for_llm(df_train, feature_name)

    # 3. Inference
    y_true = []
    y_pred = []
    
    logger.log(f"⏳ Running predictions on {len(df_val)} records...")

    for i, row in enumerate(df_val.iter_rows(named=True)):
        text_input = row['text_content']
        true_score = row[feature_name]
        
        try:
            # CALL TO LLM
            llm_response = feature_config['func'](
                client=client, 
                content=text_input, 
                few_shot_examples=few_shot_examples
            )
            
            response_json = json.loads(llm_response)
            predicted_value = response_json.get(feature_name)
            
            # Safety Casting based on feature_config
            if feature_config['type'] == 'ordinal':
                predicted_value = int(predicted_value) if predicted_value is not None else -1
                true_score = int(true_score)
            
            elif feature_config['type'] == 'continuous':
                # Float conversion for Sentiment
                predicted_value = float(predicted_value) if predicted_value is not None else 0.0
                true_score = float(true_score)

            else:
                # Categorical (String)
                predicted_value = str(predicted_value) if predicted_value is not None else "ERROR"
                true_score = str(true_score)

        except Exception as e:
            logger.warning(f"⚠️ Error in record {i}: {e}")
            if feature_config['type'] == 'ordinal': predicted_value = -1
            elif feature_config['type'] == 'continuous': predicted_value = 0.0
            else: predicted_value = "ERROR"

        y_true.append(true_score)
        y_pred.append(predicted_value)
        
        if (i+1) % 10 == 0: print(f"   Processed {i+1}/{len(df_val)}...")

    # 4. Metrics & Reporting
    logger.log("-" * 60)
    logger.log(f"📊 METRICS logger: {feature_name}")
    
    # --- ORDINAL LOGIC ---
    if feature_config['type'] == 'ordinal':
        error_value = adjacent_accuracy(y_true, y_pred) # adjacent_tol = 1 by default
        logger.log(f"   🎯 Adjacent Accuracy:  {error_value:.2%} (Target: {feature_config['validation_threshold']:.0%}) (Tolerance +/- 1)")
        
        if feature_name == 'content_relevance_score':
            cutoff = feature_config['cutoff']
            bin_true = [1 if x >= cutoff else 0 for x in y_true]
            bin_pred = [1 if x >= cutoff else 0 for x in y_pred]
            bin_acc = accuracy_score(bin_true, bin_pred)
            logger.log(f"   ⚖️ Binary Filter Acc:  {bin_acc:.2%} (Score >= {cutoff})")

    # --- CONTINUOUS LOGIC (SENTIMENT) ---
    elif feature_config['type'] == 'continuous':
        error_value = mean_absolute_error(y_true, y_pred)
        logger.log(f"   📉 Mean Absolute Error (MAE): {error_value:.4f} (Target: < {feature_config['validation_threshold']})")

    # --- CATEGORICAL LOGIC ---
    elif feature_config['type'] == 'categorical':
        error_value = accuracy_score(y_true, y_pred)
        logger.log(f"   🎯 Exact Accuracy:     {error_value:.2%} (Target: {feature_config['validation_threshold']:.0%})")

    if error_value <= feature_config['validation_threshold']:
        logger.log("   ✅ SUCCESS: Error is within acceptable limits.")
    else:
        logger.log("   🛑 FAILURE: High error rate.")

    logger.log("-" * 60)

#==============================================================================

def run_generation_for_feature(feature_name, feature_file_path, feature_config, df, df_train, batch_save_size, pilot_mode, pilot_size, pilot_seed, client, logging): 

    mode_msg = f"🧪 PILOT MODE (Max {pilot_size} records)" if pilot_mode else "🚀 PRODUCTION MODE (Full Data)"
    logging.info(f"STARTING GENERATION of {feature_name}")
    logging.info(f"MODE: {mode_msg}")

    few_shot_examples = process_labeled_sample_for_llm(df_train, feature_name)

   # 4. PREPARE DATA (Resume Logic)
    
    # A. Check what is already done
    processed_ids = set()
    if os.path.exists(feature_file_path):
        try:
            df_existing = pl.read_parquet(feature_file_path)
            processed_ids = set(df_existing['comment_id'].to_list())
            logging.info(f"🔄 Resume: Found {len(processed_ids)} records already processed in output file.")
        except Exception as e:
            logging.warning(f"⚠️ Output file exists but couldn't be read: {e}")

    # B. Filter out processed records
    df_to_process = df.filter(~ pl.col('comment_id').is_in(processed_ids))
    
    # C. Apply PILOT LIMIT (The only change in logic)
    if pilot_mode:
        if len(df_to_process) > pilot_size:
            logging.info(f"✂️ Cutting dataset to {pilot_size} records for Pilot test.")
            df_to_process = df_to_process.sample(n=pilot_size, seed=pilot_seed)
    
    n_to_process = len(df_to_process)
    if n_to_process == 0:
        logging.info("✅ No new records to process. Exiting.")
        return

    logging.info(f"⏳ Queue size: {n_to_process} new records to process.")

    # 5. PROCESSING LOOP
    results_buffer = [] 
    n_processed_records = 0
    
    for row in df_to_process.iter_rows(named=True):
        comment_id = row['comment_id']
        text_input = row['text_content']
                
        try:
            # CALL TO LLM
            llm_response = feature_config['func'](
                client=client, 
                content=text_input, 
                few_shot_examples=few_shot_examples
            )
            
            response_json = json.loads(llm_response)
            predicted_value = response_json.get(feature_name)
            
            # Safety Casting based on feature_config
            if feature_config['type'] == 'ordinal':
                predicted_value = int(predicted_value) if predicted_value is not None else -1
                true_score = int(true_score)
            
            elif feature_config['type'] == 'continuous':
                # Float conversion for Sentiment
                predicted_value = float(predicted_value) if predicted_value is not None else 0.0
                true_score = float(true_score)

            else:
                # Categorical (String)
                predicted_value = str(predicted_value) if predicted_value is not None else "ERROR"
                true_score = str(true_score)

        except Exception as e:
            logging.warning(f"⚠️ Error in record {i}: {e}")
            if feature_config['type'] == 'ordinal': predicted_value = -1
            elif feature_config['type'] == 'continuous': predicted_value = 0.0
            else: predicted_value = "ERROR"
        
        # Add result to buffer
        results_buffer.append({
            "comment_id": comment_id,
            feature_name: llm_response
        })
        
        n_processed_records += 1

        # 6. Incremental Saving (Batching)
        if n_processed_records % batch_save_size == 0 or n_processed_records == n_to_process:
            logging.info(f"💾 Saving batch... ({n_processed_records}/{n_to_process})")
            
            df_new_chunk = pl.DataFrame(results_buffer)
            
            # Append Logic
            if os.path.exists(feature_file_path):
                try:
                    df_current = pl.read_parquet(feature_file_path)
                    # Vertical concat
                    df_combined = pl.concat([df_current, df_new_chunk])
                    df_combined.write_parquet(feature_file_path)
                except Exception as e:
                    logging.error(f"❌ Error saving batch: {e}")
            else:
                # Create new file
                df_new_chunk.write_parquet(feature_file_path)
            
            # Clear buffer
            results_buffer = []

    logging.info("✅ Generation Process Completed.")

#==============================================================================

class ValidationLogger:
    """Handles logger to both console and text file."""

    def __init__(self, reports_dir, feature_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(reports_dir, f"val_report_{feature_name}_{timestamp}.txt")
        self.buffer = []
        self.log(f"VALIDATION REPORT: {feature_name.upper()} - {timestamp}")
        self.log("="*60 + "\n")

    def log(self, message):
        print(message)
        self.buffer.append(str(message))

    def save(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.buffer))
        print(f"\n✅ Report saved to: {self.filename}")

#==============================================================================
