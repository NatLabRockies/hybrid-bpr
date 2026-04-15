"""MovieLens training pipeline runner.

Usage:
    python movielens/run_movielens.py
    python movielens/run_movielens.py --dataset ml-20m --sweep
    python movielens/run_movielens.py --hero
"""

import argparse
import os

from hybridbpr import (
    TrainingPipeline, UserItemData, init_hero_mlflow,
    load_movielens_ui, FEATURE_DATASETS
)


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
        default=3.0,
        help='Min rating for positive; below this = negative'
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
    parser.add_argument(
        '--experiment',
        type=str,
        default=None,
        help='MLflow experiment base name (overrides config value)'
    )
    args = parser.parse_args()

    # Initialize training pipeline from config
    print(f"Loading config: {args.config}")
    pipeline = TrainingPipeline(config_path=args.config)

    # Append dataset name to experiment for per-dataset separation
    base_exp = args.experiment or pipeline.cfg.get(
        'mlflow.experiment_name', 'movielens'
    )
    pipeline.cfg['mlflow.experiment_name'] = (
        f'{base_exp}-{args.dataset}'
    )

    # Optionally initialize Hero MLflow backend
    custom_mlflow = None
    if args.hero:
        custom_mlflow, _ = init_hero_mlflow(pipeline)

    # item_feature comes from config; negatives always loaded
    rating_threshold = args.rating_threshold
    item_feature = pipeline.cfg.get('data.item_feature', 'metadata')

    # Build UserItemData from config defaults
    print(f"\nLoading {args.dataset} and building UserItemData...")
    ui = load_movielens_ui(
        dataset=args.dataset,
        rating_threshold=rating_threshold,
        item_feature=item_feature,
    )

    # Factory rebuilds ui when sweep changes data.* params
    def ui_factory(cfg: dict) -> UserItemData:
        """Rebuild UserItemData from per-experiment config."""
        return load_movielens_ui(
            dataset=args.dataset,
            rating_threshold=rating_threshold,
            item_feature=cfg.get('data.item_feature', item_feature),
        )

    # Run training or sweep
    mlflow_kwargs = (
        {"custom_mlflow": custom_mlflow} if custom_mlflow else {}
    )
    all_run_ids = pipeline.run(
        ui,
        sweep=args.sweep,
        num_processes=args.n_jobs,
        ui_factory=ui_factory,
        **mlflow_kwargs
    ) or []


if __name__ == '__main__':
    main()
