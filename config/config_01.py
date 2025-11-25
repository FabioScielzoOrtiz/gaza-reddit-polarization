# --- API LIMITS AND TIME FILTERS ---
MAX_LIMIT = 5 # 1000 
TIME_FILTER = 'year' 

# --- SEARCH STRATEGIES ---

# Subreddits configuration list. All subreddits are now implicitly configured 
# for full post AND comment extraction.
LIST_SUBREDDITS = [
    # Pro-Palestinian/Left-Leaning Communities
    #'Palestine',
    #'Gaza',
    #'SocialDemocracy',
    'Liberal',
    
    # General Political Forums
    'politics',
    
    # Niche Ideological/Identity Groups
    #'JewsOfConscience',
    #'DemocraticSocialism',
    
    # Discussion/Q&A Forums - Liberal Polarity
    #'AskALiberal', 
    
    # Pro-Israel/Conservative-Leaning Communities
    #'Israel', 
    #'Jewish', 
    'Conservative', 
    
    # Discussion/Q&A Forums - Conservative Polarity
    #'AskConservatives', 
    #'ProgressivesForIsrael', 
    
    # Neutral/General Discussion & Opinion
    #'PoliticalDiscussion', 
    #'geopolitics', 
    #'IsraelPalestine', 
    #'TrueUnpopularOpinion', 
    #'UnpopularOpinion', 
]

# PRAW Search Sort Methods.
LIST_SORTS = [
    'relevance', 
    #'top', 
    #'new', 
    #'comments'
]

# Queries representing the discourse frames
LIST_QUERIES = [
    # Q1: Humanitarian/Legal Frame
    #'((gaza OR palestine*) AND (genocide OR humanitarian* OR UN OR ICJ OR victims OR war crime)) OR (israel AND (genocide OR humanitarian* OR UN OR ICJ OR victims OR war crime))',
    
    # Q2: Conflict/Security Frame
    '((gaza OR palestine*) AND (hamas OR terroris* OR attack OR hostages OR netanyahu OR IDF OR war OR conflict)) OR (israel AND (hamas OR terroris* OR attack OR hostages OR netanyahu OR IDF OR war OR conflict))',
    
    # Q3: Geopolitical/Political Frame
    #'((gaza OR palestine*) AND (biden OR trump OR US OR congress OR EU OR ally OR antisemit* OR "anti-semit*")) OR (israel AND (biden OR trump OR US OR congress OR EU OR ally OR antisemit* OR "anti-semit*"))',
    
    # Q4: Media/Narrative Frame
    #'((gaza OR palestine*) AND (media OR narrative OR propaganda OR bias OR reporting OR antisemit* OR "anti-semit*" OR islamophob* OR misinformation OR "hate speech")) OR (israel AND (media OR narrative OR propaganda OR bias OR reporting OR antisemit* OR "anti-semit*" OR islamophob* OR misinformation OR "hate speech"))',
    
    # Q5: General Topic Catch-all
    #'(gaza OR palestine*)'
]

