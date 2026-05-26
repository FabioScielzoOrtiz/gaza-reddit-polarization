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
**CRITICAL CONTEXT: THE "DISCARD" THRESHOLD (< 4)**
Keep in mind that any comment receiving a score of 0, 1, 2 or 3 **will be completely discarded from the final analysis**. You must use these lower scores confidently to filter out any comment that does not provide useful data about the public opinion on the Gaza conflict (e.g., purely domestic US politics, generic noise, meta-Reddit discussions).

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

---
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
You are an expert political analyst working on a research study about the Gaza conflict.

Your task is to assign a Political Stance Score from 1 (Pro-Palestine) to 5 (Pro-Israel) based on the comment provided.

The metric should be measured based on the "comment_body", which is the unit of analysis; the "post_title" and "post_body" should be used to provide context for the "comment_body". 

---
**STRICT GUIDELINES:**
1. **FOCUS:** Analyze the **Comment Body**. Use Post context only for interpretation.
2. **REASONING STYLE:** The reasoning must be a very brief, single-sentence statement describing the primary action, emotion, or argument of the comment (e.g., "The comment shows empathy towards Palestine"). Do NOT explicitly name the scale labels or numbers in your reasoning text.
3. **CONTEXTUAL AGREEMENT (CRITICAL):** If a comment strongly agrees with a "Strongly Pro-Palestine" Post Title (e.g., calling people who deny genocide "ignorant"), it is a 1, NOT a 2 or a 3. The intensity of the agreement inherits the intensity of the Post.
4. **ANTI-POLARITY FLIP, SARCASM & MOCKING (CRITICAL):** Watch out for sarcasm and laughter (e.g., "Hahaha", "Lol"). 
   - If a comment mocks or laughs at a Pro-Israel post title/news to DEFEND a Palestinian slogan (e.g., arguing that 'Free Palestine' just means freedom, not violence), it is a 1 or 2. 
   - If it mocks, mimics, or laughs at Pro-Palestine protesters, slogans, or narratives to invalidate them, it is a 4 or 5. 
   - ALWAYS identify the *ultimate target* of the mockery. Laughing at an anti-Palestinian narrative is a Pro-Palestine action, and vice-versa.
5. **TONE vs STANCE:** A very polite, well-written comment that systematically defends Israeli policies, denies genocide, or focuses heavily on the hostages is a 5, not a 4 or 3. Stance is about the ARGUMENT'S DIRECTION, not how polite the user is.
6. **THE NEUTRALITY TRAP (3 vs 4):** Score 3 is EXCLUSIVE for comments that explicitly attack or defend BOTH sides symmetrically, or are cold, objective recaps of news. If a comment shows even mild skepticism towards Pro-Palestinian movements or justifies military necessity, it is a 4, NOT a 3. 
7. **THE UNCLASSIFIABLE GUARDRAIL (-1 vs 1-5 - CRITICAL):** You MUST assign -1 if the comment discusses macro-regional geopolitics, military logistics, or economic strategy (e.g., Iran's strategy in the Gulf, US military movements, oil refineries) without expressing an explicit evaluative political stance on the Gaza conflict itself. **Merely mentioning the word "Israel" or "US" as part of a list of regional targets or actors does NOT justify a 1-5 score.** If the text is purely academic, technical, or focused on other countries (like Iran, UAE, Saudi Arabia) rather than the Israeli-Palestinian dynamic, it is strictly a -1.

---
**POLITICAL STANCE SCALE:**

**1 — Strongly Pro-Palestine:** Clear contempt for IDF soldiers; unequivocally standing against Israel (e.g., stating Israel cannot exist); aggressively mocking Israeli narratives; agreeing with claims of Israeli genocide.

**2 — Leaning Pro-Palestine:** Implicit criticism of the media's treatment of Palestine; showing empathy towards Palestine civilians; sarcastic support of the Palestinian cause; pointing out statements by public figures against Palestinians.

**3 — Neutral / Balanced:** Explicit, symmetric criticism of BOTH sides OR purely factual/procedural content with zero discernible lean. Do not use this as a "dumping ground" for polite comments.

**4 — Leaning Pro-Israel:** Showing some criticism toward Palestinians, Hamas, or Pro-Palestine protests; expressing disagreement with pro-Palestinian policies; presenting nuanced arguments that ultimately place more responsibility on the Palestinian side. 

**5 — Strongly Pro-Israel:** Explicit support for Israel/IDF actions; pointing out the diffusion of Palestinian terrorism; emphasizing the need for Israeli military security; mocking pro-Palestinian narratives; or explicitly defending there is no evidence of genocide.

**-1 — Unclassifiable:** Spam, off-topic content, generic technical/legal questions, pure meta-reddit comments, isolated personal insults, or content where a political stance on the Gaza conflict itself cannot be meaningfully inferred without guessing.

---
**OUTPUT FORMAT:**
Return a single JSON object. 
Keys:
- "reasoning_political_stance_score": A brief, single-sentence explanation of what the comment expresses or focuses on.
- "political_stance_score": The integer score (1-5, -1).

---
**EXAMPLES:**

- Score -1 (Unclassifiable):
{{
  "reasoning_political_stance_score": "The comment asks a generic technical question about an international incident without expressing a stance on the Gaza conflict",
  "political_stance_score": -1
}}

- Score 1 (Strongly Pro-Palestine):
{{
  "reasoning_political_stance_score": "The comment gently insults people who do not see the genocide",
  "political_stance_score": 1
}}

- Score 2 (Leaning Pro-Palestine):
{{
  "reasoning_political_stance_score": "The comment shows empathy towards Palestine",
  "political_stance_score": 2
}}

- Score 3 (Neutral / Balanced):
{{
  "reasoning_political_stance_score": "The comment tries to be neutral in the analysis separating politicians from the general situation",
  "political_stance_score": 3
}}

- Score 4 (Leaning Pro-Israel):
{{
  "reasoning_political_stance_score": "The comment shows some criticism toward Palestinians and Hamas",
  "political_stance_score": 4
}}

- Score 5 (Strongly Pro-Israel):
{{
  "reasoning_political_stance_score": "The comment defends there is no evidence of genocide in Palestine",
  "political_stance_score": 5
}}

---
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

The metric should be measured based on the "comment_body", which is the unit of analysis; the "post_title" and "post_body" should be used to provide context for the "comment_body". 

---
**CATEGORIES (Choose exactly ONE — see Priority Rule below):**

1. Analytical: Objective tone grounded in logic, strategic reasoning, or structured argumentation. May cite sources or historical precedent. Low emotional charge. The author is building a case, not merely reporting.

2. Emotional: Dominant expression of sadness, grief, fear, empathy, or despair. Focus on human suffering. Emotional language clearly outweighs any analytical or informational content.

3. Hostile: Aggressive, insulting, dehumanizing, or ad-hominem. Directed at users, groups, or nations. Includes slurs, threats, and rhetoric that degrades the target.

4. Sarcastic: Uses irony, mockery, or satire. The intended meaning is the opposite of the literal text. The sarcastic register must be the dominant mode, not an isolated phrase.

5. Informative: Neutral sharing of links, data, breaking news, or clarifications. No argument is being constructed and no emotional stance is evident — the author's role is courier, not analyst.

6. Other: Use ONLY for content that genuinely resists classification: very short reactions (e.g. "lol", "+1", single emojis), fully off-topic comments, or incoherent text. Explain why in the reasoning field.

---
**HOW TO DETERMINE THE DOMINANT TONE (THE 2-STEP RULE):**

When a comment exhibits features of multiple categories, you MUST follow these two steps in order:

**STEP 1: Quantitative & Register Dominance (Volume & Focus)**
Identify which tone characterizes the vast majority of the text's volume, core argument, and structural intent. 
- If a comment spends 80% of its length building a logical, data-driven, or geographical case (Analytical) but ends with a single mocking punchline or insult (Sarcastic/Hostile), the dominant tone is **Analytical (1)**. The single phrase does NOT hijack the overall register.
- If a comment is short but its primary purpose is to deliver an insult or a sarcastic joke, then that tone is dominant.

**STEP 2: Tie-Breaking Hierarchy (Only for truly equal 50/50 splits)**
ONLY if two or more tones are equally balanced in weight, volume, and importance throughout the text, apply this strict tie-breaking hierarchy — assign the category highest on this list:

  Hostile (3) > Sarcastic (4) > Emotional (2) > Analytical (1) > Informative (5) > Other (6)
  
---
**OUTPUT FORMAT:**
Return a single JSON object. 
Keys:
- "reasoning_discourse_tone_score": A concise explanation (1-2 sentences) linking the text to specific scale criteria.
- "discourse_tone_score": The integer score (1-6).

---
**EXAMPLES:**

// Score 1 - Analytical
{{
  "reasoning_discourse_tone_score": "The user constructs a structured historical argument citing empires and wars to justify a geopolitical position, with no overt emotional language.",
  "discourse_tone_score": 1
}}

// Score 2 - Emotional
{{
  "reasoning_discourse_tone_score": "The comment centers on the grief of families displaced in Gaza, using language of despair and loss throughout with no analytical framing.",
  "discourse_tone_score": 2
}}

// Score 3 - Hostile
{{
  "reasoning_discourse_tone_score": "The comment uses dehumanizing language directed at a national group and includes a direct personal attack on another user.",
  "discourse_tone_score": 3
}}

// Score 4 - Sarcastic
{{
  "reasoning_discourse_tone_score": "The comment sarcastically praises a military decision ('great move, very humane') to ridicule it. The ironic register dominates the entire message.",
  "discourse_tone_score": 4
}}
{{
  "reasoning_discourse_tone_score": "The comment uses an absurd, logically impossible personal anecdote involving a celebrity to mock the post's thesis. This weaponized irony makes the sarcastic register completely dominant.",
  "discourse_tone_score": 4
}}

// Score 5 - Informative
{{
  "reasoning_discourse_tone_score": "The comment shares a Reuters article link with a one-line neutral summary and no editorial opinion or emotional framing.",
  "discourse_tone_score": 5
}}

// Score 6 - Other
{{
  "reasoning_discourse_tone_score": "The comment consists of a single emoji with no accompanying text, making tone classification unreliable.",
  "discourse_tone_score": 6
}}

---
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

Your task is to identify the **Dominant Frame** used in the text. This is the primary "lens" through which the content approaches the issue — not the topic, but the angle, emphasis, and implicit question being answered.

The metric should be measured based on the "comment_body", which is the unit of analysis; the "post_title" and "post_body" should be used to provide context for the "comment_body". Classify based on the comment_body alone — the post context provides background but should not override the frame of the comment itself.

---
**CATEGORIES (Choose exactly ONE):**

1. **Humanitarian**
   Core question: *"What is happening to people?"*
   Signals: civilian casualties, displacement, famine, hospitals, grief, aid delivery, suffering of non-combatants. The focus is on human experience and loss, without requiring a legal or accountability framework.
   ⚠️ NOT Humanitarian if: the text mentions civilian harm primarily to evaluate military proportionality, justify or critique an operation, or assess tactical effectiveness. If suffering appears as a byproduct of a strategic argument, classify as Security/Military.
   ⚠️ NOT Humanitarian if: the comment is a pure emotional reaction without explicitly referencing human suffering or civilian experience as the argumentative core. Classify as Other instead.

2. **Legal**
   Core question: *"Is this legal or legitimate under international norms?"*
   Signals: ICJ rulings, ICC proceedings, Geneva Conventions, UN resolutions, war crimes accusations, genocide as a legal term, international humanitarian law (IHL), accountability mechanisms.
   📌 The term "genocide" alone does not activate this frame. It does so only when the argument revolves around its legal definition, evidentiary standards, institutional recognition, or formal accountability. Arguments that deny or invert a legal accusation also activate this frame — e.g. "this does not meet the threshold for genocide", "Hamas is the one with genocidal intent".
   📌 A comment may use political language ("political lie", "agenda") to characterize a legal claim without that making it Geopolitical. If the core argument is about whether an action meets a legal or normative standard, classify as Legal.
   🔑 KEY DISTINCTION from Security/Military: Legal asks "Is this action lawful?" Security/Military asks "Is this action strategically justified?"
   🔑 KEY DISTINCTION from Media/Narrative: Legal challenges the factual or legal basis of a claim. Media/Narrative challenges how a claim is being constructed and circulated.

3. **Security/Military**
   Core question: *"Who is the threat and how should it be countered?"*
   Signals: IDF strategy and operations, Hamas military capabilities, hostage situations, tunnel infrastructure, terrorism framing, right to self-defense, military objectives and proportionality in tactical terms.
   ⚠️ NOT Security/Military if: the text centers on grief, loss, or civilian experience as ends in themselves — with no strategic or operational argument. Mentioning combatants, hostages, or military operations does not make a text Security/Military if the emotional and argumentative register is humanitarian or purely expressive.
   🔑 KEY DISTINCTION from Humanitarian: Ask "Is the author making a point about military logic, or about human suffering?" If the former → Security/Military. If the latter → Humanitarian.

4. **Historical/Identity**
   Core question: *"What does history, faith, or identity tell us about who is right?"*
   Signals: Nakba, British Mandate, 1948/1967 wars, settlement history, land rights claims, origins of the conflict, Palestinian displacement history, biblical or Quranic justifications, "Promised Land," Jewish/Islamic identity as the primary analytical lens, divine mandate or destiny framing.
   📌 Use when the argument is grounded in historical facts or identity — rather than in present events, institutions, or strategy.

5. **Media/Narrative**
   Core question: *"How is this story being told, and who controls the narrative?"*
   Signals: media bias accusations, propaganda, disinformation, hasbara, coverage framing (CNN/BBC/Al Jazeera comparisons), discourse control, debates over which narrative dominates the public sphere.
   ⚠️ NOT Media/Narrative if: the text simply expresses a political or humanitarian opinion — even forcefully. The frame requires the text to be about the *representation* of the conflict, not the conflict itself.

6. **Geopolitical**
   Core question: *"What are political actors doing and why?"*
   Signals: international alliances, US/Iran/Egypt/Qatar roles, UNSC vetoes, diplomatic pressure, ceasefire negotiations, arms supply decisions, electoral incentives, governing coalitions, parliamentary debates, public opinion pressure, political survival of leaders, campus protests as domestic political events.
   📌 The argument must be about what political actors are doing, deciding, or failing to do — not merely about whether a political entity should exist or is morally justified.
   ⚠️ NOT Geopolitical if: the comment is a short question or reaction with no argument about political agency or actors.
   🔑 KEY DISTINCTION from Ideological: Geopolitical describes or evaluates what political actors do. Ideological declares what justice, rights, or moral order require — without engaging with political agency or process.

7. **Ideological**
   Core question: *"What does justice require here?"*
   Signals: claims about legitimacy or illegitimacy of states or movements, colonial or apartheid framing, right to resistance, Zionism as ideology, moral equivalences, justice framing, declarations about who is right or wrong in the conflict — without grounding in legal standards, historical argument, or political analysis.
   📌 This frame captures articulated ideological positions that go beyond pure emotion but fall short of analytical argument. The author is declaring how the world should be, not describing how it is.
   ⚠️ NOT Ideological if: the position is grounded in a legal framework (→ Legal), a historical argument (→ Historical/Identity), or an analysis of political actors (→ Geopolitical).
   🔑 KEY DISTINCTION from Geopolitical: Ideological says "this is wrong/unjust". Geopolitical says "this is what actor X is doing and why".
   🔑 KEY DISTINCTION from Legal: Legal engages with evidentiary or normative standards. Ideological bypasses them — "scholars aside, common sense dictates" is Ideological, not Legal.

8. **Other**
   Use only when the text is genuinely unclassifiable — i.e., it does not meaningfully invoke any of the above frames. This includes fully off-topic comments, incoherent text, or content that are just pure emotions with non ideological or geopolitical arguments. If you are tempted to use it, re-read the eight categories above carefully.

---
**CLASSIFICATION RULES:**
- Choose the ONE frame that best captures the dominant lens of the text.
- If multiple frames appear, select the one that is most structurally central to the argument or emotional register — not merely mentioned in passing.
- Do not confuse *topic* with *frame*: a text about civilian deaths can be Humanitarian (suffering focus), Legal (accountability focus), or Security/Military (collateral damage as tactical reality), depending on how it is framed.
- The presence of violence, casualties, or military actors alone does not determine the frame — the *argumentative purpose* does.
- Classify based on the comment_body alone. Post context helps interpret ambiguous cases but must not override the frame explicitly present in the comment.
- "Other" should be rare. If you are tempted to use it, re-read the category descriptions carefully.

---
**OUTPUT FORMAT:**
Return a single JSON object.
Keys:
- "reasoning_dominant_frame_score": A concise explanation (1-2 sentences) linking specific textual signals to the chosen category's core question.
- "dominant_frame_score": The integer score (1-9).

---
**Example outputs:**

// 1. Humanitarian
{{
    "reasoning_dominant_frame_score": "The text foregrounds civilian suffering — child hunger, destroyed hospitals, generational harm — without invoking legal accountability or military logic. The register is emotional and centered on human experience.",
    "dominant_frame_score": 1
}}

// 2. Legal
{{
    "reasoning_dominant_frame_score": "The argument revolves around whether Israel's actions meet the legal threshold for genocide under the Genocide Convention — the central question is evidentiary and legal, not emotional or strategic.",
    "dominant_frame_score": 2
}}

// 3. Security/Military
{{
    "reasoning_dominant_frame_score": "The text analyzes tunnel infrastructure, IDF tactical options, and the military objective of neutralizing Hamas — the lens is purely strategic, focused on threat assessment and operational effectiveness. Although civilian deaths are mentioned, they appear as part of a proportionality argument, not as the emotional core.",
    "dominant_frame_score": 3
}}

// 4. Historical/Identity
{{
    "reasoning_dominant_frame_score": "The argument explicitly roots the present conflict in the Nakba and 1948 displacement, treating historical land rights and foundational events as the essential frame for understanding the current situation.",
    "dominant_frame_score": 4
}}

// 5. Media/Narrative
{{
    "reasoning_dominant_frame_score": "The text is a meta-analysis of media framing — comparing outlet language and arguing that news coverage shapes rather than reflects the conflict. No primary analytical frame about the conflict itself is present.",
    "dominant_frame_score": 5
}}

// 6. Geopolitical
{{
    "reasoning_dominant_frame_score": "The text centers on Biden's electoral calculus and donor pressure — the primary question is what political actors are doing and why, not the moral legitimacy of their positions.",
    "dominant_frame_score": 6
}}

// 7. Ideological
{{
    "reasoning_dominant_frame_score": "The comment declares that Zionism is a colonial project and that resistance is therefore legitimate — an articulated ideological position about justice and legitimacy, without grounding in legal standards or political analysis.",
    "dominant_frame_score": 7
}}

// 8. Other
{{
    "reasoning_dominant_frame_score": "The comment is fully off-topic and contains no reference to the conflict or any analytical frame. It is genuinely unclassifiable.",
    "dominant_frame_score": 9
}}

---
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
You are an academic researcher evaluating the quality of public deliberation about the Gaza conflict.

Your task is to assign an **Argument Quality Score** from **0 to 5** based on the sophistication, structure, and justification of the text — regardless of the political position it defends.

The metric should be measured based on the "comment_body", which is the unit of analysis; the "post_title" and "post_body" should be used to provide context for the "comment_body". 

---
**SCORING RUBRIC:**

**0 — Spam / Non-Argument**
The text cannot be evaluated as deliberation. Includes: broken or garbled text, bot-generated noise, purely phatic content ("lol", "???"), or content with no discernible communicative intent related to the conflict.

**1 — Pure Reaction**
A position or emotion is expressed but no reason is given. Includes: slogans ("Free Palestine", "I stand with Israel"), name-calling, single-word reactions, emoji-only responses, or content whose entire argumentative value is an assertion of alignment.

**2 — Bare Opinion**
A position is stated with at least a minimal justification, but the reasoning is thin, circular, or purely repetitive of talking points. The claim-to-evidence link is weak or absent. May include a single undeveloped assertion ("Israel has the right to defend itself because Hamas attacked") without elaboration.

**3 — Justified Opinion**
A position supported by at least one coherent, developed reason — whether empirical, moral, or experiential. The logic is followable even if not rigorous. May rely on a single line of argument. Personal testimony counts if it is used to support a claim, not just to express feeling.
Key question: *Does the text build a case, however simple?*

**4 — Reasoned Argument**
A structured argument that links claims to evidence or reasoning in more than one step. Shows contextual awareness: acknowledges complexity, qualifies claims, references specific events/data/actors, or anticipates an implicit objection. Does not require formal citation — a well-constructed analytical paragraph qualifies.

**5 — Sophisticated Discourse**
Exceptional deliberative quality. Combines several of: explicit engagement with counter-arguments, citation of specific legal frameworks/sources/empirical data, synthesis of multiple causal or normative dimensions, epistemic humility about uncertainty, or original analytical framing. This is rare in public discourse — reserve it for texts that would be defensible in an academic or high-quality journalistic context.

---
**SCORING RULES:**
- Score the **argumentative quality**, not the political position. A sophisticated pro-Israel argument scores the same as a sophisticated pro-Palestinian one with equivalent structure.
- Score what is **present in the text**, not what you infer the author knows. An expert writing a single slogan scores 1.
- Emotional language does not penalize a score if it accompanies reasoning. Emotion alone, without reasoning, caps the score at 1.
- Length does not determine quality. A long repetitive rant scores 2; a single sharp analytical sentence can score 4.
- When in doubt between two adjacent scores, choose the lower one.
- **EMOTIONAL LANGUAGE DOES NOT EQUAL PURE REACTION (CRITICAL):** Do not penalize a comment or cap it at 1 just because it uses high-intensity emotional words (e.g., "starving", "horrors", "sickening"). If the emotional text establishes a causal mechanism (e.g., "X horror is happening now, therefore Y future outcome is impossible") or provides a baseline reason for a stance, it MUST be scored as **2 (Bare Opinion)** or **3 (Justified Opinion)**. Emotion *accompanied* by a logical link or implication is a valid form of public deliberation.
- **LOGICAL BREVITY & IMPLICIT ARGUMENTS:** Redditors often use short, dense statements or rhetorical questions to convey a complete logical point (e.g., a firsthand personal experience that invalidates a post's premise, or an unbacked empirical example). If a human reader can map a clear "Premise -> Conclusion" link from the text without guessing the user's mind, score it based on that logical structure (2 or 3), not on its length or lack of academic jargon.

---
**OUTPUT FORMAT:**
Return a single JSON object.
Keys:
- "reasoning_argument_quality_score": A concise explanation (1-2 sentences) identifying the specific textual features that determine the score — reference rubric criteria explicitly.
- "argument_quality_score": The integer score (0-5).

---
**EXAMPLES:**

// Score 0 — Spam / Non-Argument
{{
    "reasoning_argument_quality_score": "The text is garbled, contains no discernible communicative intent, and offers no argument or position that can be evaluated.",
    "argument_quality_score": 0
}}

// Score 1 — Pure Reaction
{{
    "reasoning_argument_quality_score": "The text expresses strong alignment and uses charged language, but offers no reason, evidence, or developed claim — it is pure stance declaration.",
    "argument_quality_score": 1
}}

// Score 2 — Bare Opinion
{{
    "reasoning_argument_quality_score": "A clear position is stated and a minimal justification is given (October 7th as trigger event), but the reasoning is a single undeveloped assertion and the rhetorical question substitutes for argument rather than advancing one.",
    "argument_quality_score": 2
}}

// Score 3 — Justified Opinion
{{
    "reasoning_argument_quality_score": "The text builds a causal argument linking prolonged blockade conditions to radicalization, using a specific timeline (2007, 16 years) to support the claim. The logic is coherent and developed, though it does not engage counter-arguments or cite sources.",
    "argument_quality_score": 3
}}

// Score 4 — Reasoned Argument
{{
    "reasoning_argument_quality_score": "The text applies a specific legal concept (IHL proportionality) correctly, distinguishes it from common misuse, acknowledges a counter-consideration (human shields), and explicitly flags a logical fallacy in public debate — demonstrating multi-step reasoning and contextual sophistication.",
    "argument_quality_score": 4
}}

// Score 5 — Sophisticated Discourse
{{
    "reasoning_argument_quality_score": "The text demonstrates mastery of the relevant legal doctrine (dolus specialis, ICJ procedure), correctly describes the January ruling, identifies an underappreciated conceptual distinction (legal vs. moral-political registers), and does so with explicit epistemic charity toward both sides — meeting the threshold for sophisticated discourse.",
    "argument_quality_score": 5
}}

---
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
You are an expert in Natural Language Processing (NLP) specializing in sentiment analysis of political discourse on social media.

Your task is to analyze the **Emotional Valence** of the text regarding the Gaza conflict.
Assign a continuous **Sentiment Score** from **-1.0** to **1.0** based exclusively on the tone and emotional register of the language — not on the political position expressed or its moral validity.

The metric should be measured based on the "comment_body", which is the unit of analysis; the "post_title" and "post_body" should be used to provide context for the "comment_body". 

---
**SCORING RUBRIC:**

**-1.0 to -0.7 — Very Negative**
Extreme emotional intensity in the negative register. Includes: dehumanizing language, hate speech, explicit calls for violence/destruction, graphic expressions of rage, unmitigated contempt, or deep trauma/despair directed outwards or inwards without any element of communal solidarity.

**-0.6 to -0.4 — Moderately Negative**
Consistent negative tone without reaching extremity. Includes: sustained criticism, moral condemnation, cynical commentary, frustration, or sarcasm. The text may be analytical, but the emotional coloring is clearly critical or negative throughout.

**-0.3 to -0.1 — Mildly Negative**
A slight negative lean, but largely restrained. Includes: measured concern, cautious skepticism, pure empathetic grief (condolences), understated disappointment, or factual reporting with a critical undertone.

**0.0 — Neutral**
No detectable emotional loading. Purely descriptive or factual statements, objective questions, dry technical/military evaluations, or observations that balance opposing emotional signals to net zero.

**0.1 to 0.3 — Mildly Positive**
A slight positive lean. Includes: understated solidarity, cautious hope, mild praise, or empathetic but measured language. The tone is warmer than neutral but not demonstratively optimistic.

**0.4 to 0.6 — Moderately Positive**
Consistent positive tone. Includes: clear expressions of support, hope, encouragement, or agreement. The emotional register is affirmative without reaching intense celebration or deep collective catharsis.

**0.7 to 1.0 — Very Positive**
Strong positive emotional intensity. Includes: intense celebration, deep gratitude, fervent expressions of pride, love, or relief, and highly solemn vows of communal solidarity or defense of memory. The text is emotionally saturated in the positive or supportive direction.

---
**SCORING RULES (CRITICAL NLP GUARDRAILS):**

- **The "In-Group Solidarity vs. Out-Group Hate" Principle:** Do not mistake fierce communal alignment, solidarity, or protective vows for negative rage. Mantras of memory and absolute resolve like "Never forget and never forgive" or "Am Yisrael Chai" when posted in reaction to a victim's testimony represent maximum positive alignment, conviction, and solemn group support **(0.7 to 1.0)**. They are expressions of protective unity, not unmitigated negative hate.
- **The "Tribute Effect" (Commemorative Grief):** In texts celebrating positive communal milestones (e.g., hostages returning), the mention of crying for those lost or honoring the dead does NOT reduce or penalize the positive score. Commemorative grief increases the solemnity and emotional saturation of the support. If the text uses highly elevated positive terms ("beyond ecstatic") alongside mourning for the group's losses, score it at the extreme positive end **(0.7 to 1.0)**.
- **The "Empathetic Grief" Trap (Negative vs. Positive):** Do not confuse pro-social intent with positive emotional valence. Expressions of pure sorrow, condolences, and empathy for victims (e.g., "This is heartbreaking," "My condolences") represent the emotion of grief and sadness. Even though the intent is supportive, the emotional register is intrinsically negative. Score pure empathetic grief in the negative band **(-0.1 to -0.3)**, NOT in the positive band.
- **Technical Praise vs. Emotional Positivity:** Do not confuse the technical evaluation of military/institutional performance with emotional positivity. Phrases like "exceptional air defense" or "strategic win" are often part of a dry, analytical assessment of security, not expressions of joy or warmth. If the text primarily evaluates tactics or strategic effectiveness without clear emotional language, score it closer to **Neutral (0.0)** or slightly negative if it emphasizes vulnerabilities.
- **Score the tone, not the content.** A factual description of atrocities is not automatically Very Negative — a detached clinical report may score near 0.0.
- Return scores to **one decimal place** (e.g., -0.7, 0.4, 0.0, 1.0).

---
**OUTPUT FORMAT:**
Return a single JSON object.
Keys:
- "reasoning_sentiment_score": A concise explanation (1-2 sentences) identifying the specific lexical features and applying the Guardrails to justify the score band.
- "sentiment_score": A float between -1.0 and 1.0 (one decimal place).

---
**EXAMPLES:**

- 1.0 to -0.7 — Very Negative
{{
    "reasoning_sentiment_score": "Dehumanizing language and absolute moral contempt with no restraint — the text is saturated with unmitigated hate and sits at the extreme negative end.",
    "sentiment_score": -0.9
}}

- -0.3 to -0.1 — Mildly Negative (Empathetic Grief)
{{
    "reasoning_sentiment_score": "The text expresses deep empathy and offers condolences ('this is heartbreaking'). Under the Empathetic Grief rule, pure sorrow and mourning constitute a restrained negative emotional valence, despite the pro-social intent.",
    "sentiment_score": -0.1
}}

- 0.0 — Neutral (Technical Praise)
{{
    "reasoning_sentiment_score": "The comment analytically praises military interception rates ('exceptional air defense') but focuses on costs and tactical limitations. This is a dry technical evaluation lacking true emotional positivity, netting a neutral score.",
    "sentiment_score": 0.0
}}

- 0.7 to 1.0 — Very Positive (Solidarity & Resolve)
{{
    "reasoning_sentiment_score": "The text uses emphatic mantras of memory ('Never forget and never forgive') as a solemn vow of communal solidarity and fierce defensive alignment, placing it in the extreme positive support band.",
    "sentiment_score": 0.8
}}

- 0.7 to 1.0 — Very Positive (Tribute Effect)
{{
    "reasoning_sentiment_score": "The text combines intense celebratory terms ('beyond ecstatic', 'glorious day') with commemorative grief for the fallen. Under the Tribute Effect, this mourning amplifies the emotional saturation of group solidarity rather than penalizing it.",
    "sentiment_score": 1.0
}}

---
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

def grouped_accuracy_political_stance(y_true, y_pred):
    """
    Accuracy agrupada para political_stance_score:
    - {1, 2} → Pro-Palestine
    - {4, 5} → Pro-Israel
    - {3}    → Neutral (exact match only)
    - {-1}   → Unclassifiable (exact match only)
    """
    def to_group(val):
        if val in (1, 2):
            return 'pro_palestine'
        elif val in (4, 5):
            return 'pro_israel'
        elif val == 3:
            return 'neutral'
        elif val == -1:
            return 'unclassifiable'
        else:
            return 'unknown'
    
    y_true_grouped = [to_group(v) for v in y_true]
    y_pred_grouped = [to_group(v) for v in y_pred]
    
    return accuracy_score(y_true_grouped, y_pred_grouped)

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
            llm_reasoning = response_json.get(f"reasoning_{feature_name}")


            # Safety Casting (consistente con el runner de generación)
            if feature_type != 'continuous':
                predicted_value = int(predicted_value) if predicted_value is not None else -1
                true_value = int(true_value)
            else:
                predicted_value = float(predicted_value) if predicted_value is not None else 0.0
                true_value = float(true_value)

            return {
                'comment_id': comment_id,
                'true_value': true_value,
                'predicted_value': predicted_value,
                'reasoning': llm_reasoning,
                'raw_response': llm_response
            }

        except Exception as e:
            logging.warning(f"⚠️ Error in record {comment_id}: {e}")
            fallback_val = -1 if feature_type == 'ordinal' else (0.0 if feature_type == 'continuous' else "ERROR")
            return {
                'comment_id': comment_id,
                'true_value': true_value,
                'predicted_value': fallback_val,
                'reasoning': llm_reasoning,
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
    iterations_predicted_values, iterations_reasoning = [], []
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
        iterations_reasoning.append(
            [r['reasoning'] for r in all_iter_results]
        )

        if feature_type == 'ordinal':
            validation_results['individual_validation']['score_type'] = 'accuracy'
            if feature_name != 'content_relevance_score':
                score_value = accuracy_score(y_true, y_pred) #adjacent_accuracy(y_true, y_pred) 
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
            
            if feature_name == 'political_stance_score':
                score_value = grouped_accuracy_political_stance(y_true, y_pred)
                logging.info(f"  🎯 Iter {iter_idx} - Grouped Accuracy: {score_value:.2%}")
            else:
                score_value = accuracy_score(y_true, y_pred)
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
    validation_results['llm_metadata']['iterations_reasoning'] = {
        c: [x[c_idx] for x in iterations_reasoning] 
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
    # FIX 1: memory_map=False on the resume read so the file handle is released immediately
    processed_ids = set()
    if os.path.exists(feature_file_path):
        try:
            df_existing = pl.read_parquet(feature_file_path, memory_map=False)
            processed_ids = set(df_existing['comment_id'].to_list())
            del df_existing  # explicit release
            logging.info(f"🔄 Resume: Found {len(processed_ids)} processed records.")
        except Exception:
            pass

    df_to_process = df.filter(~pl.col('comment_id').is_in(processed_ids))
    
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
        # FIX 2: memory_map=False on the concat read + FIX 3: write to tmp then rename
        if results:
            df_new_chunk = pl.DataFrame(results)
            tmp_path = feature_file_path + ".tmp"
            try:
                if os.path.exists(feature_file_path):
                    df_current = pl.read_parquet(feature_file_path, memory_map=False)
                    merged = pl.concat([df_current, df_new_chunk])
                    del df_current  # release before write
                    merged.write_parquet(tmp_path)
                    del merged
                else:
                    df_new_chunk.write_parquet(tmp_path)
                
                # Atomic replace: delete original, rename tmp
                if os.path.exists(feature_file_path):
                    os.remove(feature_file_path)
                os.rename(tmp_path, feature_file_path)
                
                logging.info(f"💾 Batch {i}-{min(i+batch_save_size, total_records)} saved.")

            except Exception as e:
                logging.error(f"❌ Error saving batch: {e}")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)  # clean up orphaned tmp file

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
