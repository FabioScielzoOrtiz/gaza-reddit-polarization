##########################################################################################################################################################

import json
import os
import logging

##########################################################################################################################################################

# Configure logging at the module level
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

##########################################################################################################################################################

def process_expert_samples(project_path: str) -> dict:
    """
    Loads all expert sample files from the 'data/expert_samples' directory.
    Handles File Not Found and JSON decoding errors safely.
    """
    expert_samples_dir = os.path.join(project_path, 'data', 'expert_sample')
    
    if not os.path.exists(expert_samples_dir):
        logging.warning(f"❌ Expert samples directory not found: {expert_samples_dir}. Returning empty samples.")
        return {}
    
    expert_samples = {}
    
    # 1. Iterate through all files in the samples directory
    for filename in os.listdir(expert_samples_dir):
        if filename.endswith('.json'):
            sample_name = filename.split('.')[0]
            file_path = os.path.join(expert_samples_dir, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    expert_samples[sample_name] = json.load(f)
                logging.info(f"✅ Loaded expert samples for: {sample_name}")
                
            except FileNotFoundError:
                logging.error(f"❌ File not found: {file_path}. Skipping.")
                expert_samples[sample_name] = []
            except json.JSONDecodeError as e:
                # Critical fix: Handle malformed JSON specifically
                logging.error(f"❌ JSON Decode Error in {filename}: {e}. Skipping.")
                expert_samples[sample_name] = []
            except Exception as e:
                # Catch any other unexpected I/O errors
                logging.error(f"❌ Unexpected error loading {filename}: {e}. Skipping.")
                expert_samples[sample_name] = []
                
    return expert_samples


##########################################################################################################################################################

# Load expert samples
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
expert_samples = process_expert_samples(project_path)

##########################################################################################################################################################

def content_relevance_score(client, content):
        
    prompt = f"""
You are a content rating specialist for an academic study on public opinion regarding the Gaza conflict on Reddit.

Your task is to assign a numerical **Relevance Score** from **0 (Not Related)** to **5 (Directly Related)** to the provided text (Post Title + Post Body + Comment Body).

---
**STRICT GUIDELINE:** 1. **FOCUS:** The score MUST primarily reflect the relevance of the **Comment Body** as an opinion regarding the Gaza conflict. Use the Post Title/Body ONLY as context to understand the comment and the topic.
2. **SCORE CRITERIA:** **DO NOT** penalize the text based on its tone (aggressive, polemical), quality (low effort, vague), or brevity. The only criterion is whether the text explicitly discusses the Israel-Palestine/Gaza conflict or its direct context.

---
**TOPICAL RELEVANCE SCALE (0-5):**

* **5 - Directly Related (Core Focus):** The comment's main subject is the Israel-Palestine/Gaza conflict. It explicitly mentions the conflict, parties (Israel, Palestine, Hamas, IDF), or core related events/politics. High score regardless of the text's tone or quality.
* **4 - Clearly Related (Strong Context):** The comment mentions the conflict, but the discussion is brief, a simple reaction, or a very aggressive/polemical statement. It is unambiguously about the conflict.
* **3 - Marginal Context:** The comment uses related keywords (e.g., "Middle East," "foreign policy," "UN") but does not explicitly mention "Gaza," "Israel," or "Palestine." The connection to the conflict is implied, not stated.
* **2 - Accidental/Trivial Mention:** The comment uses a keyword (e.g., "Gaza") but is clearly talking about something else (e.g., "a recipe from Gaza") or is pure noise (e.g., a simple "Thank you" in a related thread).
* **1 - Off-Topic Noise/Abuse:** The comment is primarily an attack on another user (*ad hominem*) or a simple emotional outburst *without* mentioning the conflict. Pure noise that is not topical.
* **0 - Discard/Spam:** The comment is solely spam, broken text, or completely unrelated content.

---
**OUTPUT FORMAT:**
You must output a single JSON object. Do not include any reasoning, salutations, or additional text.

**EXAMPLE OUTPUT:**
{{"content_relevance_score": 4}}

**TEXT TO CLASSIFY:**
{content}
"""

    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a political analyst and content classifier. Your output must be a single JSON object based ONLY on the provided criteria."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
            temperature=0.1
        )
    return response.choices[0].message.content


##########################################################################################################################################################

def political_stance_score(client, content):
    
    sample_key = 'political_stance_samples'
    expert_samples_formatted = ""
    expert_samples_list = expert_samples.get(sample_key)
    if expert_samples_list:
        # Format the loaded expert samples to inject in the prompt
        expert_samples_formatted = json.dumps(expert_samples_list, indent=2)
    else:
        logging.warning("Expert samples for political_stance is empty. That expert knowledge will not be able to be used by the LLM to generate the data.")
            
        
    prompt = f"""
You are an expert political analyst for an academic study on public opinion regarding the Gaza conflict.

Your task is to assign a single **Political Stance Score** from **1 (Strongly Pro-Palestine)** to **5 (Strongly Pro-Israel)** to the provided text.

---
**STRICT GUIDELINE:** 1. **FOCUS:** The score MUST reflect the political alignment expressed in the **Comment Body**. Use the Post Title/Body ONLY as neutral context.
2. **TONE:** Base the score on the **underlying political position**, not the tone (aggressive/polemical comments can still be scored 1 or 5).
3. **NEUTRALITY:** Only assign a score of **3** if the comment explicitly criticizes or supports BOTH sides equally, or if the discussion is purely academic/historical without taking a side.

---
**POLITICAL STANCE SCALE (1-5):**
* 5 - Strongly Pro-Israel: Explicit, strong support for Israeli actions...
* 4 - Mildly Pro-Israel: Support for Israel's right to self-defense...
* 3 - Neutral/Balanced: Objective discussion, historical analysis...
* 2 - Mildly Pro-Palestine: Focus on the humanitarian crisis...
* 1 - Strongly Pro-Palestine: Explicit, strong condemnation of Israeli actions...

---
**EXPERT KNOWLEDGE: REFERENCE SAMPLES**
The following samples are derived from expert human ratings. **You MUST use these examples as the definitive standard** to determine the expected score for each category (1-5):
{expert_samples_formatted}

---
**OUTPUT FORMAT:**
You must output a single JSON object. Do not include any reasoning, salutations, or additional text.

**EXAMPLE OUTPUT:**
{{"political_stance": 2}}

**TEXT TO CLASSIFY:**
{content}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a political analyst and content classifier. Your output must be a single JSON object based ONLY on the provided criteria."},
            {"role": "user", "content": prompt}
        ],
        response_format={ "type": "json_object" },
        temperature=0.1
    )
    return response.choices[0].message.content

##########################################################################################################################################################

