
import os
import polars as pl 

script_path = os.path.dirname(os.path.abspath(__file__)) 
project_path = os.path.join(script_path, '..')
processed_data_dir = os.path.join(project_path, 'data', 'processed_data')
processed_data_path = os.path.join(processed_data_dir, '06_processed_data.parquet')
processed_data = pl.read_parquet(processed_data_path)

embeddings_cols_50 = sorted(
    [col for col in processed_data.columns if col.startswith("embedding_pca50_")]
)

embeddings_cols_90 = sorted(
    [col for col in processed_data.columns if col.startswith("embedding_pca90_")]
)

clust_config_metadata = {

    'clust_config_I': {
        'quant_cols':  ['sentiment_score'] # numerical_cols
                    +  ['argument_quality_score', 'political_stance_score'], # ordinal_cols
        'binary_cols': [],
        'multiclass_cols': ['discourse_tone_score', 'dominant_frame_score'], # nominal_cols         
        'n_clusters': 4, 
        'kmedoids_method': 'pam',
        'metric': 'ggower',
        'd1': 'robust_mahalanobis',
        'd2': 'sokal',
        'd3': 'hamming',
        'robust_method': 'trimmed',
        'alpha': 0.05,
        'frac_sample_size': 0.15,
        'random_state': 123
    },

    'clust_config_I_b': {
        'quant_cols':  ['sentiment_score'] # numerical_cols
                    +  ['argument_quality_score', 'political_stance_score'], # ordinal_cols
        'binary_cols': [],
        'multiclass_cols': ['discourse_tone_score', 'dominant_frame_score'], # nominal_cols         
        'n_clusters': 4,
        'random_state': 123  
    },

    'clust_config_II': {
        'quant_cols':  ['sentiment_score'] # numerical_cols
                    +  embeddings_cols_50,
        'binary_cols': [],
        'multiclass_cols': ['discourse_tone_score', 'dominant_frame_score'] # nominal_cols    
                        +  ['argument_quality_score', 'political_stance_score'], # ordinal_cols    
        'n_clusters': 4, 
        'kmedoids_method': 'pam',
        'metric': 'ggower',
        'd1': 'robust_mahalanobis',
        'd2': 'sokal',
        'd3': 'hamming',
        'robust_method': 'trimmed',
        'alpha': 0.05,
        'frac_sample_size': 0.15,
        'random_state': 123
    },

    'clust_config_III': {
        'quant_cols':  embeddings_cols_50,
        'binary_cols': [],
        'multiclass_cols': [],
        'n_clusters': 4,   
        'random_state': 123     
    },

    'clust_config_III_b': {

        'quant_cols': embeddings_cols_90,
        'binary_cols': [],
        'multiclass_cols': [],
        'n_clusters': 4,
        'random_state': 123
    },

}
