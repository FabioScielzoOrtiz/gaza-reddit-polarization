# gaza-reddit-polarization

Code for the article *"Quantifying polarization: Detection and analysis of opinion groups on the Gaza conflict in English-language Reddit discourse"* (Scielzo-Ortiz, Grané and Díaz-Gorfinkiel, submitted to the *Journal of Computational Social Science*).

The pipeline collects Reddit comments on the Gaza conflict, extracts five discursive variables with an LLM (validated against expert-coded samples), builds semantic embeddings, and benchmarks five clustering configurations, selecting the final partition with a Pairwise Separability Index. Full documentation of every stage is given in Online Resource 1 (Sect. S5) of the article.

## Setup

- Python 3.12
- `pip install -r requirements.txt`
- Create a `.env` file in the repository root with your own credentials (it is not versioned):

```
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
OPENAI_API_KEY=...
```

## Pipeline

Scripts in `src/scripts/` are run in numeric order from the repository root; each one reads the Parquet file written by the previous stage (in `data/`, not versioned) and writes a new one. Stage parameters live in `config/`.

| Stage | Scripts | Output |
|---|---|---|
| 01–02 | Data extraction (PRAW) and processing | Unified comment table |
| 03a–03d | Relevance score: labelling sample, validation, generation, filtering | Relevance-filtered corpus |
| 04a–04d | Five discursive variables: labelling sample, validation, generation, merge | Discursive variables |
| 05a–05c | Embeddings (`text-embedding-3-large`) and PCA (τ = 0.50, 0.90) | Embedding components |
| 06a | Analytical sample preparation | `06_processed_data.parquet` (n = 76,816) |
| 07a (notebook) | Candidate partitions for k ∈ {2,3,4,5} | Diagnostics for choosing k |
| 07b | Fit the five k = 4 configurations | `models/*.joblib` |
| 07c (notebook) | Profiles, c-TF-IDF, Pairwise Separability Index, silhouette | Figures and tables of the article |
| `stability_experiment_config_I.py` | 30-seed ARI stability of Configuration I | `data/stability_results/` |

LLM calls use `gpt-4o-mini` at temperature 0. Running stages 03–05 requires OpenAI API access and incurs costs.

## Data availability

Raw Reddit content cannot be redistributed under the platform's terms of use. Comment identifiers, LLM-derived variables and cluster assignments are available from the corresponding author on reasonable request.

## Citation

If you use this code, please cite the article above.
