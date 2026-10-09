#################################################################################################

# ANALYTICAL SAMPLE AUDIT
# Documents how the relevance-filtered corpus becomes the n = 76,816 analytical sample:
#   (1) row counts per stage, duplicate relevance scores and comments without embedding;
#   (2) rows removed by each exclusion rule of 06a, applied sequentially;
#   (3) origin of each excluded value: explicit model decision (non-empty reasoning) vs.
#       extraction failure (empty reasoning -> fallback code -1 assigned by the generation script);
#   (4) extraction-failure rates by subreddit and political stance.
# Read-only with respect to the pipeline data; results are written to data/data_analysis_results.

#################################################################################################

# --- IMPORTS ---

import os
import json
import logging
import polars as pl

#################################################################################################

# --- LOGGING CONFIGURATION ---

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#################################################################################################

# --- PATH CONFIGURATION ---

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..')

processed_data_dir = os.path.join(project_path, 'data', 'processed_data')
features_dir = os.path.join(project_path, 'data', 'features')
results_dir = os.path.join(project_path, 'data', 'data_analysis_results')
os.makedirs(results_dir, exist_ok=True)

#################################################################################################

# --- PARAMETERS ---

STAGES = ['02_processed_data', '03d_processed_data', '04d_processed_data',
          '05c_processed_data', '06_processed_data']

# Exclusion rules of 06a, in the order in which they are applied here
EXCLUSION_RULES = [
    ('political_stance_score', [-1]),
    ('discourse_tone_score', [-1]),
    ('dominant_frame_score', [-1, 9]),
    ('argument_quality_score', [-1]),
]

FAILURE_FEATURE = 'argument_quality_score'   # feature with the largest number of failures
TOP_SUBREDDITS = 8

#################################################################################################

# --- FUNCTIONS ---

def read_stage(name, columns=None):
    return pl.read_parquet(os.path.join(processed_data_dir, f'{name}.parquet'), columns=columns)


def empty_reasoning(feature):
    col = pl.col(f'reasoning_{feature}')
    return col.is_null() | (col.str.strip_chars() == '')


def stage_counts():
    rows = []
    for name in STAGES:
        ids = read_stage(name, columns=['comment_id'])['comment_id']
        rows.append({'stage': name, 'rows': len(ids), 'unique_comment_ids': ids.n_unique()})
    return pl.DataFrame(rows)


def relevance_duplicates():
    rel = pl.read_parquet(os.path.join(features_dir, 'content_relevance_score.parquet'),
                          columns=['comment_id', 'content_relevance_score'])
    dup = rel.filter(pl.col('comment_id').is_duplicated())
    conflicting = (dup.group_by('comment_id')
                      .agg(pl.col('content_relevance_score').n_unique().alias('k'))
                      .filter(pl.col('k') > 1).height)
    return {'relevance_rows': rel.height,
            'relevance_unique_ids': rel['comment_id'].n_unique(),
            'duplicated_ids': dup['comment_id'].n_unique(),
            'duplicated_ids_with_conflicting_scores': conflicting}


def missing_embeddings():
    emb = pl.read_parquet(os.path.join(features_dir, 'embeddings_pca.parquet'), columns=['comment_id'])
    d04 = read_stage('04d_processed_data', columns=['comment_id'])
    return d04.join(emb, on='comment_id', how='anti').height


def sequential_exclusions(df):
    rows, remaining = [], df
    for feature, codes in EXCLUSION_RULES:
        excluded = remaining.filter(pl.col(feature).is_in(codes))
        n_fail = excluded.filter(empty_reasoning(feature)).height
        rows.append({'rule': f'{feature} in {codes}',
                     'removed': excluded.height,
                     'extraction_failures': n_fail,
                     'model_decisions': excluded.height - n_fail,
                     'remaining': remaining.height - excluded.height})
        remaining = remaining.filter(~pl.col(feature).is_in(codes))
    return pl.DataFrame(rows), remaining


def failure_rates(df, by):
    return (df.group_by(by)
              .agg(pl.len().alias('n'), empty_reasoning(FAILURE_FEATURE).sum().alias('failures'))
              .with_columns((100 * pl.col('failures') / pl.col('n')).round(1).alias('failure_pct'))
              .sort('n', descending=True))

#################################################################################################

# --- MAIN EXECUTION ---

def main():

    # 1. Row counts per stage, duplicates and missing embeddings
    counts = stage_counts()
    dups = relevance_duplicates()
    n_no_emb = missing_embeddings()
    logging.info(f'Row counts per stage:\n{counts}')
    logging.info(f'Relevance duplicates: {dups}')
    logging.info(f'Comments of 04d without PCA embedding: {n_no_emb}')

    # 2-3. Sequential exclusions of 06a and origin of the excluded values
    df = read_stage('05c_processed_data')
    exclusions, remaining = sequential_exclusions(df)
    logging.info(f'Sequential exclusions (input n = {df.height:,}):\n{exclusions}')
    logging.info(f'Analytical sample after exclusions: {remaining.height:,}')

    # 4. Extraction-failure rates by subreddit and by political stance
    by_subreddit = failure_rates(df, 'post_subreddit').head(TOP_SUBREDDITS)
    by_stance = failure_rates(df, 'political_stance_score').sort('political_stance_score')
    logging.info(f'{FAILURE_FEATURE} failures by subreddit (top {TOP_SUBREDDITS} by size):\n{by_subreddit}')
    logging.info(f'{FAILURE_FEATURE} failures by political stance:\n{by_stance}')

    # 5. Save results
    counts.write_csv(os.path.join(results_dir, 'audit_stage_counts.csv'))
    exclusions.write_csv(os.path.join(results_dir, 'audit_sequential_exclusions.csv'))
    by_subreddit.write_csv(os.path.join(results_dir, 'audit_failures_by_subreddit.csv'))
    by_stance.write_csv(os.path.join(results_dir, 'audit_failures_by_stance.csv'))
    summary = {**dups,
               'comments_without_embedding': n_no_emb,
               'input_06a': df.height,
               'analytical_sample': remaining.height}
    with open(os.path.join(results_dir, 'audit_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    logging.info(f'📁 Audit results saved in {results_dir}.')

if __name__ == '__main__':
    main()

#################################################################################################
