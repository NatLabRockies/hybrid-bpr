# hybrid-bpr

A Python library for Bayesian Personalized Ranking (BPR) with two key
capabilities that go beyond standard BPR implementations:

1. **User and item feature embeddings** — incorporate content-based
   signals (genres, tags, metadata) alongside collaborative filtering
2. **Implicit negative interactions** — use observed non-interactions
   (e.g. viewed-but-not-clicked) as negative training signal instead
   of random sampling from the full item space

Built for recommender systems research with MLflow experiment tracking,
parallel hyperparameter sweeps, and standard ranking metrics.

---

## Why hybrid-bpr?

### Standard BPR limitation 1: no feature support

Standard BPR learns a pure latent factor per user and item with no
way to incorporate side information. `hybrid-bpr` supports sparse
feature matrices for both users and items:

- **User features** (`Fu`): user attributes, demographics, history
- **Item features** (`Fi`): genres, tags, categories, embeddings

Scores are computed as:

```
score(u, i) = (Fu[u] @ U) · (Fi[i] @ V) + bias[i]
```

This enables generalization to cold-start users and items that share
features with seen entities.

### Standard BPR limitation 2: random negative sampling

Standard BPR samples negatives uniformly at random from all unobserved
items. In large catalogs this is noisy — most random negatives are
irrelevant items the user simply never encountered, not items they
actively disliked or ignored.

`hybrid-bpr` lets you supply an **implicit negative interaction matrix**
(`Rneg`) of observed non-interactions — items a user was exposed to but
did not engage with. Negatives are sampled from this pool, giving a
much stronger and less noisy training signal:

```
Rpos: user clicked / purchased / rated highly   → positive
Rneg: user viewed / was shown / skipped         → implicit negative
```

---

## Installation

Requires Python 3.13+. Clone the repo then install with either `pip`
or `uv`:

```bash
git clone https://github.com/rimplesandhu/pybpr.git
cd pybpr

# pip
pip install -e .

# uv (faster, recommended for new environments)
uv pip install -e .
```

---

## Core API

| Class / Function        | Purpose                                      |
|-------------------------|----------------------------------------------|
| `UserItemData`          | Stores interaction and feature matrices      |
| `MatrixFactorization`   | PyTorch model scoring (user, item) pairs     |
| `RecommendationSystem`  | BPR training loop with evaluation           |
| `TrainingPipeline`      | Config-driven orchestrator + grid search    |
| `load_movielens()`      | Auto-download MovieLens datasets             |
| `MLflowPlotter`         | Visualize MLflow experiment results          |

---

## Quick Start

```python
import numpy as np
from pybpr import UserItemData, TrainingPipeline

rng = np.random.default_rng(42)
n_users, n_items = 500, 200

# Positive interactions (e.g. purchases)
pos_users = rng.integers(0, n_users, 5000)
pos_items = rng.integers(0, n_items, 5000)

# Implicit negatives (e.g. viewed but not purchased)
neg_users = rng.integers(0, n_users, 8000)
neg_items = rng.integers(0, n_items, 8000)

# Build dataset
ui = UserItemData(name="demo")
ui.add_positive_interactions(pos_users, pos_items)
ui.add_negative_interactions(neg_users, neg_items)

# Identity (one-hot) user and item features
ui.add_user_features(np.arange(n_users), np.arange(n_users))
ui.add_item_features(np.arange(n_items), np.arange(n_items))
ui.split_train_test(train_ratio=0.8)

# Train
config = TrainingPipeline.create_default_config()
config['model']['n_latent'] = 32
config['training']['n_iter'] = 100
config['mlflow']['experiment_name'] = 'demo'

pipeline = TrainingPipeline(config=config)
system = pipeline.run(ui)
```

---

## MovieLens Example

The `movielens/` directory contains a complete training pipeline using
the open-source MovieLens dataset.

### Load data

```python
from pybpr import load_movielens

# Auto-downloads ml-100k if not present
data = load_movielens('ml-100k', preprocess=True)
```

Supported datasets:

| Dataset  | Size   | Ratings    |
|----------|--------|------------|
| ml-100k  | 5 MB   | 100,000    |
| ml-1m    | 6 MB   | 1,000,000  |
| ml-10m   | 63 MB  | 10,000,000 |
| ml-20m   | 190 MB | 20,000,000 |
| ml-25m   | 250 MB | 25,000,000 |

### Item feature modes

A key strength of `hybrid-bpr` is supporting multiple item feature
representations, selectable via the `item_feature` config option:

| Mode        | Description                                        |
|-------------|----------------------------------------------------|
| `indicator` | One-hot per item (pure collaborative filtering)    |
| `metadata`  | Multi-hot genre/tag features (content-based)       |
| `both`      | Concatenated indicator + metadata (hybrid)         |

For MovieLens, `metadata` mode encodes the 19 genre indicators as
item features, enabling recommendations that generalize across genres.

### Implicit negatives in MovieLens

Ratings below the threshold (default 3.5) are treated as implicit
negatives — the user watched the movie but did not rate it highly.
This is a much stronger signal than randomly sampling unwatched movies.

```bash
cd movielens

# Single run, ratings >= 3.5 are positive, lower ratings are negative
python run_movielens.py --dataset ml-100k --rating-threshold 3.5

# Grid search across latent dims and feature types
python run_movielens.py --dataset ml-100k --sweep --n-jobs 4
```

### Config (`movielens/config.yaml`)

```yaml
model:
  n_latent: 64
  init_std: 0.1

optimizer:
  name: Adam
  lr: 0.001
  weight_decay: 0.0001

training:
  loss_function: bpr_loss
  n_iter: 3000
  batch_size: 100
  eval_every: 10
  top_k: 20
  early_stopping_patience: 1000

data:
  item_feature: indicator   # indicator | metadata | both
  train_ratio_pos: 0.8

mlflow:
  experiment_name: movielens-runs
  tracking_uri: sqlite:///mlflow.db

sweep:
  model.n_latent: [32, 64, 128]
  data.item_feature: [metadata, indicator, both]
```

---

## Loss Functions

All losses operate on pairwise scores `r_ui` (positive) and `r_uj`
(negative):

| Name          | Formula                                         |
|---------------|-------------------------------------------------|
| `bpr_loss`    | `-log(sigmoid(σ * (r_ui - r_uj)))`             |
| `bpr_loss_v2` | `-log(sigmoid(r_ui)) - log(sigmoid(-r_uj))`    |
| `hinge_loss`  | `ReLU(margin - (r_ui - r_uj))`                 |
| `warp_loss`   | rank-weighted hinge                             |

---

## Evaluation Metrics

Computed on held-out test interactions per user and logged to MLflow:

- **AUC** — pairwise ranking quality
- **NDCG@k** — normalized discounted cumulative gain
- **Precision@k** — fraction of top-k that are relevant
- **Recall@k** — fraction of relevant items in top-k

---

## Hyperparameter Grid Search

```python
from pybpr import TrainingPipeline

config = TrainingPipeline.create_default_config()
config['mlflow']['experiment_name'] = 'sweep'

pipeline = TrainingPipeline(config=config)

results = pipeline.run_grid_search(
    ui,
    param_grid={
        'model.n_latent': [32, 64, 128],
        'optimizer.lr': [0.001, 0.01],
        'training.loss_function': ['bpr_loss', 'hinge_loss'],
    },
    num_processes=4,
)
```

All 12 combinations run in parallel with results logged to MLflow.

---

## MLflow Tracking

```bash
mlflow ui --backend-store-uri sqlite:///movielens/mlflow.db
```

Open http://localhost:5000 to compare runs.

---

## Citation

```bibtex
@software{hybrid_bpr,
    title={hybrid-bpr: Feature-Rich BPR with Implicit Negatives},
    author={Rimple Sandhu and Charles Tripp},
    year={2024},
    url={https://github.com/rimplesandhu/pybpr}
}
```

Original BPR paper:

> Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L.
> (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback.
> *UAI 2009*.

---

## Contact

- **Rimple Sandhu** — rimple.sandhu@nlr.gov
- **Charles Tripp** — charles.tripp@nlr.gov

National Laboratory of the Rockies (NLR)
