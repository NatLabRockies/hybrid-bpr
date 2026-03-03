"""
Simple Training Pipeline Example

Demonstrates minimal code to use TrainingPipeline.
Shows separation between data prep and model training.
Reuse same UserItemData with different training configs.
"""

import numpy as np

from pybpr import UserItemData, TrainingPipeline


def create_synthetic_data():
    """Create a simple synthetic dataset for demonstration."""
    # Generate synthetic interactions
    np.random.seed(42)
    n_users = 1000
    n_items = 500
    n_interactions = 10000

    # Random user-item pairs
    user_ids = np.random.randint(0, n_users, size=n_interactions)
    item_ids = np.random.randint(0, n_items, size=n_interactions)

    # Create UserItemData
    ui = UserItemData(name='synthetic_dataset')

    # Add positive interactions
    ui.add_positive_interactions(
        user_ids=user_ids, item_ids=item_ids
    )

    # Add user features (identity)
    ui.add_user_features(
        user_ids=np.arange(n_users),
        feature_ids=np.arange(n_users)
    )

    # Add item features (identity)
    ui.add_item_features(
        item_ids=np.arange(n_items),
        feature_ids=np.arange(n_items)
    )

    print(f"Created synthetic dataset: {ui}")
    return ui


def example_1_basic_training():
    """Example 1: Basic training with default config."""
    print("="*80)
    print("Example 1: Basic Training with Default Config")
    print("="*80)

    # Prepare data
    ui = create_synthetic_data()

    # Create default config and run
    config = TrainingPipeline.create_default_config()
    config['training']['n_iter'] = 50
    config['mlflow']['experiment_name'] = 'simple_pipeline_examples'

    pipeline = TrainingPipeline(config=config)
    run_ids = pipeline.run(ui)
    print(f"\nMLflow run ID: {run_ids[0]}")


def example_2_config_file():
    """Example 2: Training with YAML config file."""
    print("\n" + "="*80)
    print("Example 2: Training with YAML Config File")
    print("="*80)

    # Prepare data
    ui = create_synthetic_data()

    # Train using config file (mlflow settings come from the file)
    pipeline = TrainingPipeline(config_path='examples/config.yaml')
    run_ids = pipeline.run(ui)
    print(f"\nMLflow run ID: {run_ids[0]}")


def example_3_custom_params():
    """Example 3: Training with custom parameters."""
    print("\n" + "="*80)
    print("Example 3: Training with Custom Parameters")
    print("="*80)

    # Prepare data
    ui = create_synthetic_data()

    # Create custom config
    config = TrainingPipeline.create_default_config()
    config['model']['n_latent'] = 128
    config['training']['n_iter'] = 100
    config['training']['loss_function'] = 'hinge_loss'
    config['optimizer']['lr'] = 0.001
    config['mlflow']['experiment_name'] = 'simple_pipeline_examples'

    pipeline = TrainingPipeline(config=config)
    run_ids = pipeline.run(ui)
    print(f"\nMLflow run ID: {run_ids[0]}")


def example_4_grid_search():
    """Example 4: Hyperparameter grid search."""
    print("\n" + "="*80)
    print("Example 4: Hyperparameter Grid Search")
    print("="*80)

    # Prepare data
    ui = create_synthetic_data()

    # Create config
    config = TrainingPipeline.create_default_config()
    config['training']['n_iter'] = 30
    config['mlflow']['experiment_name'] = 'example4_grid_search'

    # Define parameter grid
    param_grid = {
        'model.n_latent': [32, 64],
        'optimizer.lr': [0.001, 0.01],
        'training.loss_function': ['bpr_loss', 'hinge_loss']
    }

    # Run grid search
    pipeline = TrainingPipeline(config=config)
    results = pipeline.run_grid_search(
        ui,
        param_grid=param_grid,
        mlflow_experiment_name='example4_grid_search',
        base_run_name='grid_search',
        num_processes=2
    )

    print(f"\nCompleted {len(results)} experiments")


def example_5_multiple_datasets():
    """Example 5: Train same model on multiple datasets."""
    print("\n" + "="*80)
    print("Example 5: Same Config, Multiple Datasets")
    print("="*80)

    # Create multiple datasets
    datasets = []
    for i in range(3):
        np.random.seed(i)
        ui = create_synthetic_data()
        ui.name = f'dataset_{i}'
        datasets.append(ui)

    # Use same pipeline for all datasets
    config = TrainingPipeline.create_default_config()
    config['training']['n_iter'] = 30
    config['mlflow']['experiment_name'] = 'example5_multiple_datasets'
    pipeline = TrainingPipeline(config=config)

    # Train on each dataset
    for ui in datasets:
        print(f"\nTraining on {ui.name}...")
        run_ids = pipeline.run(ui)
        print(f"MLflow run ID: {run_ids[0]}")


if __name__ == '__main__':
    import sys

    # Map example numbers to functions
    examples = {
        1: example_1_basic_training,
        2: example_2_config_file,
        3: example_3_custom_params,
        4: example_4_grid_search,
        5: example_5_multiple_datasets
    }

    # Run specific example or default to example 1
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        if example_num in examples:
            examples[example_num]()
        else:
            print(
                f"Example {example_num} not found. "
                f"Available: 1-5"
            )
    else:
        # Run example 1 by default
        example_1_basic_training()
