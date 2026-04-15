"""Zazzle training pipeline runner.

Usage:
    python zazzle/run_zazzle.py
    python zazzle/run_zazzle.py --event-type orders --sweep
"""

import argparse
import os

import numpy as np

from hybridbpr import TrainingPipeline, UserItemData, init_hero_mlflow
from hybridbpr.zazzle import EVENT_TYPES, load_zazzle


def build_user_item_data(
    data: dict,
    item_feature: str,
    name: str,
    use_negatives: bool = True,
) -> UserItemData:
    """Build UserItemData from Zazzle data dict."""
    pos = data['positives']
    neg = data['negatives']
    fdf = data['features']

    # For metadata/both: filter to products with feature info
    if item_feature in ('metadata', 'both'):
        known = set(fdf['ProductID'].unique())
        pos = pos[pos['ProductID'].isin(known)]
        neg = neg[neg['ProductID'].isin(known)]

    # Initialize UserItemData
    ui = UserItemData(name=name)

    # Add positive interactions; optionally add explicit negatives
    ui.add_positive_interactions(
        user_ids=pos['UserID'].values,
        item_ids=pos['ProductID'].values
    )
    if use_negatives:
        ui.add_negative_interactions(
            user_ids=neg['UserID'].values,
            item_ids=neg['ProductID'].values
        )

    # Add user features (identity mapping) - all users incl. neg-only
    all_users = np.unique(np.concatenate([
        pos['UserID'].values, neg['UserID'].values
    ]))
    ui.add_user_features(
        user_ids=all_users,
        feature_ids=all_users
    )

    # Add item features based on configured option
    unique_products = fdf['ProductID'].unique()
    if item_feature == 'metadata':
        # Department + product type + vision style features
        ui.add_item_features(
            item_ids=fdf['ProductID'].values,
            feature_ids=fdf['FeatureID'].values
        )
    elif item_feature == 'indicator':
        # One-hot identity feature per product - all items incl. neg-only
        all_products = np.unique(np.concatenate([
            pos['ProductID'].values, neg['ProductID'].values
        ]))
        ui.add_item_features(
            item_ids=all_products,
            feature_ids=all_products
        )
    elif item_feature == 'both':
        # Metadata + indicator; offset indicator IDs above metadata
        offset = int(fdf['FeatureID'].max()) + 1
        ui.add_item_features(
            item_ids=np.concatenate(
                [fdf['ProductID'].values, unique_products]
            ),
            feature_ids=np.concatenate([
                fdf['FeatureID'].values,
                offset + unique_products
            ])
        )
    else:
        raise ValueError(f"Unknown item_feature: {item_feature}")

    print(ui)
    return ui


def main() -> None:
    """Run Zazzle training pipeline."""
    parser = argparse.ArgumentParser(
        description='Train Zazzle using TrainingPipeline'
    )
    _here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        '--config',
        type=str,
        default=os.path.join(_here, 'config.yaml'),
        help='Training config YAML path'
    )
    parser.add_argument(
        '--event-type',
        type=str,
        default='clicks',
        choices=EVENT_TYPES,
        help=(
            'clicks: clicked=pos, viewed-not-clicked=neg  |  '
            'orders: purchased=pos, clicked-not-ordered=neg'
        )
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

    # Append event type to experiment for per-event separation
    base_exp = pipeline.cfg.get('mlflow.experiment_name', 'zazzle')
    pipeline.cfg['mlflow.experiment_name'] = (
        f'{base_exp}-{args.event_type}'
    )

    # Optionally initialize Hero MLflow backend
    custom_mlflow = None
    if args.hero:
        custom_mlflow, _ = init_hero_mlflow(pipeline)

    # item_feature and use_negatives come from config
    item_feature = pipeline.cfg.get('data.item_feature', 'indicator')
    use_negatives = pipeline.cfg.get('data.use_negatives', True)

    # Load Zazzle data once (factory reuses this across sweep expts)
    print(f"Loading Zazzle data (event_type={args.event_type})...")
    data = load_zazzle(event_type=args.event_type)

    # Build UserItemData from config defaults
    print("\nBuilding UserItemData...")
    ui = build_user_item_data(
        data=data,
        item_feature=item_feature,
        name=f'zazzle_{args.event_type}_{item_feature}',
        use_negatives=use_negatives,
    )

    # Factory rebuilds ui when sweep changes data.* params
    def ui_factory(cfg: dict) -> UserItemData:
        """Rebuild UserItemData from per-experiment config."""
        feat = cfg.get('data.item_feature', item_feature)
        use_neg = cfg.get('data.use_negatives', True)
        return build_user_item_data(
            data=data,
            item_feature=feat,
            name=f'zazzle_{args.event_type}_{feat}',
            use_negatives=use_neg,
        )

    # Run training or sweep (MLflow handled by pipeline)
    mlflow_kwargs = (
        {"custom_mlflow": custom_mlflow} if custom_mlflow else {}
    )
    pipeline.run(
        ui, sweep=args.sweep, num_processes=args.n_jobs,
        ui_factory=ui_factory, **mlflow_kwargs
    )


if __name__ == '__main__':
    main()
