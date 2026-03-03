"""MovieLens training pipeline runner.

Usage:
    python movielens/run_movielens.py
    python movielens/run_movielens.py --dataset ml-20m --sweep
    python movielens/run_movielens.py --hero
"""

import argparse
import os

import numpy as np
import pandas as pd

from pybpr import TrainingPipeline, UserItemData, load_movielens

# Datasets with item features available for training:
#   ml-100k  : 19 genre indicator features (preprocessed)
#   ml-10m   : user-applied text tags (factorized to int IDs)
#   ml-20m/25m: genome relevance scores (structured tagId + weight)
FEATURE_DATASETS = ['ml-100k', 'ml-10m', 'ml-20m', 'ml-25m']


def _preprocess_ml10m(raw: dict) -> dict:
    """Normalize ml-10m text tags → integer feature IDs."""
    # Rename rating columns to match standard format
    rdf = raw['ratings'].copy()
    rdf.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

    # Factorize free-text tags to unique integer IDs per movie
    tdf = raw['tags'][['movieId', 'tag']].drop_duplicates()
    tdf = tdf.copy()
    tdf['TagID'] = pd.factorize(tdf['tag'])[0]
    tdf = tdf.rename(columns={'movieId': 'MovieID'})[
        ['MovieID', 'TagID']
    ]

    # Keep only movies with tag features
    rdf = rdf[rdf.MovieID.isin(tdf.MovieID.unique())].copy()
    return {'ratings': rdf, 'features': tdf}


def _preprocess_mlgenome(
    raw: dict,
    relevance_threshold: float = 0.5
) -> dict:
    """Normalize ml-20m/25m genome scores → feature IDs + weights."""
    # Rename rating columns to match standard format
    rdf = raw['ratings'].copy()
    rdf.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

    # Filter genome scores by relevance and rename columns
    gdf = raw['genome_scores'].copy()
    gdf = gdf[gdf['relevance'] > relevance_threshold]
    tdf = gdf.rename(
        columns={'movieId': 'MovieID', 'tagId': 'TagID'}
    )[['MovieID', 'TagID', 'relevance']]

    # Keep only movies with genome features
    rdf = rdf[rdf.MovieID.isin(tdf.MovieID.unique())].copy()
    return {'ratings': rdf, 'features': tdf}


def load_data(dataset: str) -> dict:
    """Load and normalize dataset to {'ratings', 'features'} format.

    All formats normalize ratings to UserID/MovieID/Rating columns
    and features to MovieID/TagID (+ optional relevance) columns.
    """
    if dataset == 'ml-100k':
        # load_movielens with preprocess=True returns normalized format
        return load_movielens(dataset='ml-100k', preprocess=True)
    elif dataset == 'ml-10m':
        raw = load_movielens(dataset='ml-10m', preprocess=False)
        return _preprocess_ml10m(raw)
    else:
        # ml-20m, ml-25m: use genome scores as structured features
        raw = load_movielens(dataset=dataset, preprocess=False)
        return _preprocess_mlgenome(raw)


def build_user_item_data(
    data: dict,
    rating_threshold: float,
    item_feature: str,
    name: str
) -> UserItemData:
    """Build UserItemData from normalized data dict."""
    rdf = data['ratings']
    tdf = data['features']

    # Use relevance as feature weight if present, else uniform
    weights = (
        tdf['relevance'].values
        if 'relevance' in tdf.columns
        else None
    )

    # Initialize UserItemData
    ui = UserItemData(name=name)

    # Add positive interactions (ratings at or above threshold)
    pos_mask = rdf.Rating >= rating_threshold
    ui.add_positive_interactions(
        user_ids=rdf.UserID[pos_mask].values,
        item_ids=rdf.MovieID[pos_mask].values
    )

    # Add negative interactions (ratings below threshold)
    neg_mask = rdf.Rating < rating_threshold
    ui.add_negative_interactions(
        user_ids=rdf.UserID[neg_mask].values,
        item_ids=rdf.MovieID[neg_mask].values
    )

    # Add user features (identity mapping)
    unique_users = rdf.UserID.unique()
    ui.add_user_features(
        user_ids=unique_users,
        feature_ids=unique_users
    )

    # Add item features based on configured option
    unique_movies = tdf.MovieID.unique()
    if item_feature == 'metadata':
        # Tag/genre-based semantic features
        ui.add_item_features(
            item_ids=tdf.MovieID.values,
            feature_ids=tdf.TagID.values,
            feature_weights=weights
        )
    elif item_feature == 'indicator':
        # One-hot identity features per movie
        ui.add_item_features(
            item_ids=unique_movies,
            feature_ids=unique_movies
        )
    elif item_feature == 'both':
        # Combine metadata + identity; offset indicator IDs
        offset = int(tdf.TagID.max()) + 1
        ui.add_item_features(
            item_ids=np.concatenate(
                [tdf.MovieID.values, unique_movies]
            ),
            feature_ids=np.concatenate([
                tdf.TagID.values,
                offset + unique_movies
            ]),
            feature_weights=(
                np.concatenate([weights, np.ones(len(unique_movies))])
                if weights is not None else None
            )
        )
    else:
        raise ValueError(f"Unknown item_feature: {item_feature}")

    print(ui)
    return ui


def _init_hero_mlflow(
    pipeline: 'TrainingPipeline',
) -> tuple:
    """Init Hero client and configure MLflow; return (mlflow, registry)."""
    import hero
    from dotenv import load_dotenv
    load_dotenv()

    print("Initializing Hero ML Model Registry...")
    hero_client = hero.HeroClient()
    model_registry = hero_client.MLModelRegistry()
    mlflow = model_registry.get_patched_mlflow()
    mlflow.set_tracking_uri(model_registry.get_tracking_uri())
    print(f"MLflow Tracking URI: {model_registry.get_tracking_uri()}\n")

    # Create or retrieve the experiment via Hero registry
    experiment_name = pipeline.cfg.get('mlflow.experiment_name')
    print(f"Creating/getting experiment: {experiment_name}")
    experiment = model_registry.read_or_create_experiment(experiment_name)
    print(f"Experiment ID: {experiment.experiment_id}\n")

    return mlflow, model_registry


def main() -> None:
    """Run MovieLens training pipeline."""
    parser = argparse.ArgumentParser(
        description='Train MovieLens using TrainingPipeline'
    )
    _here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        '--config',
        type=str,
        default=os.path.join(_here, 'config.yaml'),
        help='Training config YAML path'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='ml-100k',
        choices=FEATURE_DATASETS,
        help=(
            'MovieLens dataset variant '
            f'(only those with features: {FEATURE_DATASETS})'
        )
    )
    parser.add_argument(
        '--rating-threshold',
        type=float,
        default=3.5,
        help='Min rating to count as a positive interaction'
    )
    parser.add_argument(
        '--sweep',
        action='store_true',
        help='Run parameter sweep instead of single training'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=None,
        help='Parallel workers for sweep (default: all CPUs)'
    )
    parser.add_argument(
        '--hero',
        action='store_true',
        help='Log to Hero ML Model Registry instead of local MLflow'
    )
    args = parser.parse_args()

    # Initialize training pipeline from config
    print(f"Loading config: {args.config}")
    pipeline = TrainingPipeline(config_path=args.config)

    # Optionally initialize Hero MLflow backend
    custom_mlflow = None
    if args.hero:
        custom_mlflow, _ = _init_hero_mlflow(pipeline)

    # Load and normalize dataset
    print(f"Loading {args.dataset} data...")
    data = load_data(args.dataset)

    # item_feature comes from config; rating_threshold from CLI arg
    rating_threshold = args.rating_threshold
    item_feature = pipeline.cfg.get('data.item_feature', 'metadata')

    # Build UserItemData
    print("\nBuilding UserItemData...")
    ui = build_user_item_data(
        data=data,
        rating_threshold=rating_threshold,
        item_feature=item_feature,
        name=f'{args.dataset}_{item_feature}'
    )

    # Run training or sweep
    run_ids = pipeline.run(
        ui,
        sweep=args.sweep,
        num_processes=args.n_jobs,
        **({"custom_mlflow": custom_mlflow} if custom_mlflow else {})
    )

    # Plot single run and save figure (local MLflow only)
    if not args.sweep and run_ids and not args.hero:
        import matplotlib.pyplot as plt
        from pybpr import MLflowPlotter

        os.makedirs('figs', exist_ok=True)
        plotter = MLflowPlotter(
            tracking_uri=pipeline.cfg['mlflow.tracking_uri']
        )
        fig = plotter.plot_single_run(
            run_id=run_ids[0],
            figsize=(14, 5),
            std_width=2.0,
            show_std=True
        )
        save_path = f'figs/single_run_{run_ids[0]}.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
        plt.close(fig)


if __name__ == '__main__':
    main()
