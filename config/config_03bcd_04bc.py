import os, sys

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
sys.path.insert(0, project_path)

from src.feature_engineering_utils import (content_relevance_score,
                                           political_stance_score, 
                                           sentiment_score, 
                                           discourse_tone_score,
                                           dominant_frame_score,
                                           argument_quality_score
                                           )

FEATURE_CONFIG = {
    'content_relevance_score': {
        'func': content_relevance_score,
        'type': 'ordinal', # 0-5
        'cutoff': 3,       # For binary filtering check
        'validation_threshold': 0.8 # binary accuracy threshold
    },
    'political_stance': {
        'func': political_stance_score,
        'type': 'ordinal',  # 1-5
        'validation_threshold': 0.9 # adjacent accuracy threshold
    },
    'argument_quality_score': {
        'func': argument_quality_score,
        'type': 'ordinal',  # 0-5
        'validation_threshold': 0.9 # adjacent accuracy threshold
    },
    'sentiment_score': {
        'func': sentiment_score,
        'type': 'continuous', # Float -1.0 to 1.0
        'validation_threshold': 0.25 # MAE threshold
    },
    'discourse_tone': {
        'func': discourse_tone_score,
        'type': 'categorical', # Nominal (String)
        'validation_threshold': 0.8 # accuracy threshold
    },
    'dominant_frame': {
        'func': dominant_frame_score,
        'type': 'categorical', # Nominal (String)
        'validation_threshold': 0.8 # accuracy threshold
    }
}