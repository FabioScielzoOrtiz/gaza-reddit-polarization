import json
import logging
from openai import OpenAI

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

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
        logging.error(f"Error in OpenAI API call (Relevance): {e}")
        return json.dumps({"content_relevance_score": None})


# ==============================================================================
# 2. POLITICAL STANCE SCORE (Análisis Complejo)
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
        logging.error(f"Error in OpenAI API call (Stance): {e}")
        return json.dumps({"political_stance": None})