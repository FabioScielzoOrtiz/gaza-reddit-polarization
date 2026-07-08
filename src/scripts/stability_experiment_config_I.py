#################################################################################################
# STABILITY EXPERIMENT FOR CONFIG I
#################################################################################################

# --- IMPORTS ---

import os
import sys
import logging
import itertools
import numpy as np
import polars as pl
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from kmedoids import KMedoids
from db_robust_clust.models import SampleDistClustering

#################################################################################################

# --- LOGGING CONFIGURATION ---

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#################################################################################################

# --- PATH CONFIGURATION ---
# Mirrors the path logic of the original training script (assumed to live in the same
# `scripts` folder as 02_train_models.py; adjust `project_path` if this file lives elsewhere).

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..')

processed_data_dir = os.path.join(project_path, 'data', 'processed_data')
processed_data_path = os.path.join(processed_data_dir, '06_processed_data.parquet')

results_dir = os.path.join(project_path, 'data', 'stability_results')
os.makedirs(results_dir, exist_ok=True)

sys.path.append(project_path)
from config.config_07b import clust_config_metadata

#################################################################################################

# --- EXPERIMENT PARAMETERS ---

CONFIG_KEY = 'clust_config_I'
N_REPS = 30                     # number of independent sampling-seed repetitions
SAMPLING_SEEDS = list(range(1000, 1000 + N_REPS))  # arbitrary distinct seeds, disjoint from
                                                    # the original random_state=123 used in
                                                    # the main analysis, to avoid confusion
                                                    # with the primary fitted model

#################################################################################################

# --- MAIN EXECUTION ---

def main():

    # 1. Load Data
    try:
        processed_data = pl.read_parquet(processed_data_path)
        logging.info(f"Base dataset loaded: {len(processed_data)} records.")
    except Exception as e:
        logging.error(f"❌ Failed to load processed data: {e}")
        exit()

    config = clust_config_metadata[CONFIG_KEY]

    QUANT_COLS = config['quant_cols']
    BINARY_COLS = config['binary_cols']
    MULTICLASS_COLS = config['multiclass_cols']
    N_CLUSTERS = config['n_clusters']
    KMEDOIDS_METHOD = config['kmedoids_method']
    FRAC_SAMPLE_SIZE = config['frac_sample_size']
    METRIC = config['metric']
    D1 = config['d1']
    D2 = config['d2']
    D3 = config['d3']
    ROBUST_METHOD = config['robust_method']
    ALPHA = config['alpha']
    RANDOM_STATE = config['random_state']

    p1 = len(QUANT_COLS)
    p2 = len(BINARY_COLS)
    p3 = len(MULTICLASS_COLS)

    X = processed_data.select(QUANT_COLS + BINARY_COLS + MULTICLASS_COLS)
    n = X.shape[0]
    logging.info(f"Features configured. Shape of X: {X.shape}. n = {n}.")

    # 2. Fit Config I n_reps times, varying only the sampling seed
    all_labels = np.zeros((N_REPS, n), dtype=np.int64)

    for i, seed in enumerate(SAMPLING_SEEDS):

        logging.info(f"▶️ Fitting {CONFIG_KEY} | rep {i+1}/{N_REPS} | sampling_seed={seed}")

        clustering_method = KMedoids(
            n_clusters=N_CLUSTERS,
            metric='precomputed',
            method=KMEDOIDS_METHOD,
            init='build',
            max_iter=100,
            random_state=RANDOM_STATE          # <-- varied across reps
        )

        clust_object = SampleDistClustering(
            clustering_method=clustering_method,
            metric=METRIC,
            frac_sample_size=FRAC_SAMPLE_SIZE,   # <-- fixed at n_S = 0.15, as in the paper
            random_state=seed,                   # <-- varied across reps (drives which units
                                                  #     fall in the active subsample S)
            stratify=False,
            p1=p1, p2=p2, p3=p3,
            d1=D1, d2=D2, d3=D3,
            robust_method=ROBUST_METHOD, alpha=ALPHA
        )

        clust_object.fit(X)
        all_labels[i, :] = clust_object.labels_

    # 3. Save raw labels for later auditing / reproducibility
    labels_df = pl.DataFrame(
        {f"seed_{seed}": all_labels[i, :] for i, seed in enumerate(SAMPLING_SEEDS)}
    )
    labels_path = os.path.join(results_dir, 'config_I_labels_per_seed.parquet')
    labels_df.write_parquet(labels_path)
    logging.info(f"📁 Raw labels saved at {labels_path}.")

    # 4. Compute pairwise ARI across all C(N_REPS, 2) partition pairs, over the full n units
    pairwise_records = []
    for i, j in itertools.combinations(range(N_REPS), 2):
        ari = adjusted_rand_score(all_labels[i, :], all_labels[j, :])
        pairwise_records.append({
            'seed_i': SAMPLING_SEEDS[i],
            'seed_j': SAMPLING_SEEDS[j],
            'ari': ari
        })

    pairwise_df = pl.DataFrame(pairwise_records)
    pairwise_path = os.path.join(results_dir, 'config_I_stability_ari_pairwise.csv')
    pairwise_df.write_csv(pairwise_path)
    logging.info(f"📁 Pairwise ARI values saved at {pairwise_path}.")

    # 5. Summary statistics
    ari_values = pairwise_df['ari'].to_numpy()
    summary = {
        'config': CONFIG_KEY,
        'n_reps': N_REPS,
        'n_pairs': len(ari_values),
        'n_S_frac': FRAC_SAMPLE_SIZE,
        'n_individuals': n,
        'ari_mean': float(np.mean(ari_values)),
        'ari_std': float(np.std(ari_values, ddof=1)),
        'ari_min': float(np.min(ari_values)),
        'ari_max': float(np.max(ari_values)),
        'ari_median': float(np.median(ari_values)),
    }

    summary_df = pl.DataFrame([summary])
    summary_path = os.path.join(results_dir, 'config_I_stability_summary.csv')
    summary_df.write_csv(summary_path)
    logging.info(f"📁 Summary statistics saved at {summary_path}.")
    logging.info(f"✅ Summary: {summary}")

    # 6. Boxplot for visual reporting
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.boxplot(ari_values, vert=True, showmeans=True)
    ax.set_ylabel("Pairwise Adjusted Rand Index (ARI)")
    ax.set_title(
        f"Config I stability across {N_REPS} sampling seeds\n"
        f"($n_S = {int(FRAC_SAMPLE_SIZE*100)}\\%$, $n$ = {n:,})"
    )
    ax.set_xticks([])
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    plot_path = os.path.join(results_dir, 'config_I_stability_boxplot.png')
    fig.savefig(plot_path, dpi=200)
    logging.info(f"📁 Boxplot saved at {plot_path}.")


if __name__ == "__main__":
    main()

#################################################################################################