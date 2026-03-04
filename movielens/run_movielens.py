"""MovieLens training pipeline runner.

Usage:
    python movielens/run_movielens.py
    python movielens/run_movielens.py --dataset ml-20m --sweep
    python movielens/run_movielens.py --hero
"""

import argparse
import os

from hybridbpr import (
    TrainingPipeline, init_hero_mlflow,
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

    # Append dataset name to experiment for per-dataset separation
    base_exp = pipeline.cfg.get(
        'mlflow.experiment_name', 'movielens'
    )
    pipeline.cfg['mlflow.experiment_name'] = (
        f'{base_exp}-{args.dataset}'
    )

    # Optionally initialize Hero MLflow backend
    custom_mlflow = None
    if args.hero:
        custom_mlflow, _ = init_hero_mlflow(pipeline)

    # item_feature comes from config; rating_threshold from CLI arg
    rating_threshold = args.rating_threshold
    item_feature = pipeline.cfg.get('data.item_feature', 'metadata')

    # Build both variants: without and with negative interactions
    print(f"\nLoading {args.dataset} and building UserItemData...")
    ui_variants = [
        load_movielens_ui(
            dataset=args.dataset,
            rating_threshold=rating_threshold,
            item_feature=item_feature,
            use_negatives=use_neg
        )
        for use_neg in (False, True)
    ]

    # Run training or sweep for each variant
    mlflow_kwargs = (
        {"custom_mlflow": custom_mlflow} if custom_mlflow else {}
    )
    all_run_ids = []
    for ui in ui_variants:
        print(f"\n--- Running variant: {ui.name} ---")
        run_ids = pipeline.run(
            ui,
            sweep=args.sweep,
            num_processes=args.n_jobs,
            **mlflow_kwargs
        )
        all_run_ids.extend(run_ids or [])

    # Plot first single run and save figure (local MLflow only)
    if not args.sweep and all_run_ids and not args.hero:
        import matplotlib.pyplot as plt
        from hybridbpr import MLflowPlotter

        os.makedirs('figs', exist_ok=True)
        plotter = MLflowPlotter(
            tracking_uri=pipeline.cfg['mlflow.tracking_uri']
        )
        fig = plotter.plot_single_run(
            run_id=all_run_ids[0],
            figsize=(14, 5),
            std_width=2.0,
            show_std=True
        )
        save_path = f'figs/single_run_{all_run_ids[0]}.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
        plt.close(fig)


if __name__ == '__main__':
    main()
