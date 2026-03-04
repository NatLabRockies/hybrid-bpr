"""Training pipeline for PyBPR recommendation system."""

import yaml
import torch
import itertools
import logging
from functools import partial
from typing import Dict, List, Any, Optional, Callable

import mlflow
from pathos.multiprocessing import ProcessPool, cpu_count

from .recommender import RecommendationSystem
from .interactions import UserItemData
from .mf import MatrixFactorization
from .losses import LossFn

# Abbreviations for sweep param keys in run names
_KEY_ABBREVS: Dict[str, str] = {
    'n_latent': 'nl',
    'item_feature': 'feat',
    'loss_function': 'loss',
    'weight_decay': 'wd',
    'batch_size': 'bs',
    'n_iter': 'ni',
    'learning_rate': 'lr',
}

# Suppress verbose MLflow/alembic migration logs
logging.getLogger("alembic").setLevel(logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)


class TrainingPipeline:
    """Generic training pipeline for recommendation systems."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """Initialize pipeline with config from path or dict."""
        # Load or set configuration
        if config_path is not None:
            raw_config = self._load_config(config_path)
        elif config is not None:
            raw_config = config
        else:
            raise ValueError(
                "Must provide either config_path or config dict"
            )

        # Store raw and flattened versions
        self.cfg_raw = raw_config
        self.cfg = self._flatten_config(raw_config)

        # Auto-discover all loss functions from LossFn
        self.loss_function_map = {
            k: getattr(LossFn, k)
            for k in vars(LossFn)
            if not k.startswith('_')
            and callable(getattr(LossFn, k))
        }

    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    @staticmethod
    def _flatten_config(config: Dict) -> Dict:
        """Flatten nested config dict to single level with dots."""
        flat = {}
        for section, values in config.items():
            if isinstance(values, dict):
                for key, val in values.items():
                    flat[f"{section}.{key}"] = val
            else:
                flat[section] = values
        return flat

    def get_loss_function(self, loss_name: str) -> Callable:
        """Get loss function by name."""
        if loss_name not in self.loss_function_map:
            available = ', '.join(self.loss_function_map.keys())
            raise ValueError(
                f"Unknown loss function: {loss_name}. "
                f"Available options: {available}"
            )
        return self.loss_function_map[loss_name]

    def get_optimizer(
        self, optimizer_name: str, **kwargs
    ) -> partial[torch.optim.Optimizer]:
        """Get optimizer by name with parameters."""
        # Map optimizer names to torch classes
        optimizer_map = {
            'Adam': torch.optim.Adam,
            'SGD': torch.optim.SGD,
            'AdamW': torch.optim.AdamW,
            'RMSprop': torch.optim.RMSprop,
        }

        # Validate optimizer name
        if optimizer_name not in optimizer_map:
            available = ', '.join(optimizer_map.keys())
            raise ValueError(
                f"Unknown optimizer: {optimizer_name}. "
                f"Available options: {available}"
            )

        return partial(optimizer_map[optimizer_name], **kwargs)

    def build_model(self, ui: UserItemData) -> MatrixFactorization:
        """Build a MatrixFactorization model from configuration."""
        return MatrixFactorization(
            n_user_features=ui.n_user_features,
            n_item_features=ui.n_item_features,
            n_latent=self.cfg['model.n_latent'],
            sparse=self.cfg['model.sparse'],
            init_std=self.cfg.get('model.init_std', 0.1),
        )

    def run(
        self,
        ui: UserItemData,
        sweep: bool = False,
        custom_mlflow: Optional[Any] = None,
        num_processes: Optional[int] = None
    ) -> List[str]:
        """Run training; sweep=True runs full grid search."""
        # Use custom mlflow if provided, otherwise use default
        mlflow_module = custom_mlflow if custom_mlflow is not None else mlflow

        # Set MLflow tracking and experiment only if custom mlflow not supplied
        if custom_mlflow is None:
            mlflow_module.set_tracking_uri(self.cfg['mlflow.tracking_uri'])
            mlflow_module.set_experiment(
                self.cfg['mlflow.experiment_name']
            )
        else:
            # For custom mlflow, only set experiment
            mlflow_module.set_experiment(
                self.cfg['mlflow.experiment_name']
            )

        # Run sweep or single training
        if sweep:
            sweep_config = self.cfg_raw.get('sweep', {})
            if not sweep_config:
                raise ValueError("sweep config is empty")

            print("\nRunning parameter sweep...")
            results = self.run_grid_search(
                ui,
                param_grid=sweep_config,
                mlflow_experiment_name=self.cfg.get(
                    'mlflow.experiment_name'
                ),
                base_run_name=ui.name,
                num_processes=num_processes,
                custom_mlflow=custom_mlflow
            )
            print(
                f"\nSweep completed: {len(results)} experiments"
            )
            return results
        else:
            print("\nTraining single model...")
            with mlflow_module.start_run(run_name=ui.name) as run:
                self.train(ui, custom_mlflow=custom_mlflow)
                print("\nTraining completed!")
                print(f"MLflow run ID: {run.info.run_id}")
                return [run.info.run_id]

    def train(
        self,
        ui: UserItemData,
        run_name: Optional[str] = None,
        custom_mlflow: Optional[Any] = None
    ) -> RecommendationSystem:
        """Train a single model using pipeline config."""
        # Use custom mlflow if provided, otherwise use default
        mlflow_module = custom_mlflow if custom_mlflow is not None else mlflow

        # Determine run name
        if run_name is None:
            run_name = ui.name

        # Log all config parameters to MLflow
        mlflow_module.log_params(self.cfg)

        # Get loss function; bind num_items for WARP
        loss_name = self.cfg['training.loss_function']
        loss_fn = self.get_loss_function(loss_name)
        if loss_name == 'warp_loss':
            loss_fn = partial(loss_fn, num_items=ui.n_items)

        # Build model
        model = self.build_model(ui)

        # Build optimizer
        optimizer_name = self.cfg['optimizer.name']
        optimizer_params = {
            k.split('.')[1]: v
            for k, v in self.cfg.items()
            if (k.startswith('optimizer.')
                and k != 'optimizer.name')
        }
        optimizer = self.get_optimizer(
            optimizer_name, **optimizer_params
        )

        # Split data into train/test
        print(f"Starting training: {run_name}", flush=True)
        train_ratio = self.cfg.get('data.train_ratio_pos', 0.8)
        ui.split_train_test(
            train_ratio=train_ratio,
            random_state=self.cfg.get('data.random_state', None)
        )

        # Build RecommendationSystem
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        recommender = RecommendationSystem(
            uidata=ui,
            model=model,
            optimizer=optimizer,
            loss=loss_fn,
            device=device,
        )

        # Train the model
        recommender.fit(
            n_iter=self.cfg['training.n_iter'],
            n_eval_items=self.cfg.get('training.n_eval_items', 100),
            batch_size=self.cfg['training.batch_size'],
            eval_every=self.cfg['training.eval_every'],
            n_eval_users=self.cfg.get('training.n_eval_users'),
            top_k=self.cfg.get('training.top_k', 10),
            early_stopping_patience=self.cfg[
                'training.early_stopping_patience'
            ],
        )

        print(f"Finished training: {run_name}", flush=True)
        return recommender

    def run_grid_search(
        self,
        ui: UserItemData,
        param_grid: Dict[str, List],
        mlflow_experiment_name: Optional[str] = None,
        base_run_name: Optional[str] = None,
        num_processes: Optional[int] = None,
        custom_mlflow: Optional[Any] = None
    ) -> List[str]:
        """Run hyperparameter grid search in parallel."""
        # Use custom mlflow if provided, otherwise use default
        mlflow_module = custom_mlflow if custom_mlflow is not None else mlflow

        # Validate param grid
        if not param_grid:
            raise ValueError(
                "param_grid cannot be empty. Provide parameter "
                "combinations for grid search."
            )

        # Set MLflow experiment
        if mlflow_experiment_name:
            mlflow_module.set_experiment(mlflow_experiment_name)

        # Generate all parameter combinations
        all_params = self._generate_param_combinations(param_grid)
        print(f"Running {len(all_params)} experiments in grid search")

        # Auto-configure multiprocessing for optimal performance
        total_cores = cpu_count()

        # Determine number of parallel processes
        if num_processes is None:
            # Auto: use all cores, limited by number of experiments
            num_processes = min(len(all_params), total_cores)
        else:
            num_processes = min(len(all_params), num_processes)

        # Auto-calculate PyTorch threads to avoid oversubscription
        torch_num_threads = max(1, total_cores // num_processes)

        # Set PyTorch threads
        torch.set_num_threads(torch_num_threads)

        print(
            f"Using {num_processes} processes × "
            f"{torch_num_threads} PyTorch threads "
            f"({total_cores} cores available)"
        )

        # Create partial function with fixed ui
        run_single = partial(
            self._run_single_experiment,
            ui=ui,
            base_run_name=base_run_name or ui.name,
            custom_mlflow=custom_mlflow
        )

        # Run experiments in parallel with progress tracking
        print("Starting experiments...", flush=True)
        with ProcessPool(nodes=num_processes) as pool:
            # Use uimap (unordered imap) for progress tracking
            results = []
            for i, result in enumerate(
                pool.uimap(run_single, all_params), 1
            ):
                results.append(result)
                print(
                    f"[{i}/{len(all_params)}] {result}",
                    flush=True
                )

        # Print summary
        print("\nGrid Search Complete!")
        success_count = sum(
            1 for r in results if r.startswith("SUCCESS")
        )
        print(
            f"Success: {success_count}/{len(results)} experiments"
        )

        return results

    def _generate_param_combinations(
        self,
        param_grid: Dict[str, List]
    ) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters from grid."""
        # Extract parameter names and values
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        # Generate all combinations
        all_combinations = []
        for values in itertools.product(*param_values):
            params = dict(zip(param_names, values))
            all_combinations.append(params)

        return all_combinations

    def _run_single_experiment(
        self,
        params: Dict[str, Any],
        ui: UserItemData,
        base_run_name: str,
        custom_mlflow: Optional[Any] = None
    ) -> str:
        """Run one experiment; returns SUCCESS/FAILED status string."""
        # Use custom mlflow if provided, otherwise use default
        mlflow_module = custom_mlflow if custom_mlflow is not None else mlflow

        run_name = "unknown"
        try:
            # Create copy of config for this experiment
            experiment_config = self.cfg_raw.copy()

            # Generate run name from parameters
            run_name_parts = [base_run_name]

            # Update experiment config with param grid values
            for param_name, param_value in params.items():
                if '.' in param_name:
                    section, key = param_name.split('.', 1)
                    if section not in experiment_config:
                        experiment_config[section] = {}
                    experiment_config[section][key] = param_value

                    # Add abbreviated key=value to run name
                    abbrev = _KEY_ABBREVS.get(key, key)
                    if key == 'loss_function' and callable(
                        param_value
                    ):
                        run_name_parts.append(
                            f"{abbrev}{param_value.__name__}"
                        )
                    else:
                        run_name_parts.append(
                            f"{abbrev}{param_value}"
                        )

            run_name = '_'.join(str(p) for p in run_name_parts)

            # Create new pipeline with experiment config
            experiment_pipeline = TrainingPipeline(
                config=experiment_config
            )

            # Start MLflow run for this experiment
            with mlflow_module.start_run(run_name=run_name):
                # Train model with MLflow run
                experiment_pipeline.train(
                    ui=ui, run_name=run_name,
                    custom_mlflow=custom_mlflow
                )

            return f"SUCCESS: {run_name}"

        except Exception as e:
            import traceback
            error_msg = (
                f"FAILED: {run_name}\n"
                f"  Params: {params}\n"
                f"  Error: {str(e)}\n"
                f"  {traceback.format_exc()}"
            )
            return error_msg

    @staticmethod
    def create_default_config() -> Dict:
        """Create a default configuration dictionary."""
        return {
            'model': {
                'n_latent': 64,
                'sparse': False,
                'init_std': 0.1,
            },
            'optimizer': {
                'name': 'Adam',
                'lr': 0.01,
                'weight_decay': 0.0
            },
            'training': {
                'loss_function': 'bpr_loss',
                'n_iter': 100,
                'batch_size': 1000,
                'eval_every': 5,
                'n_eval_users': None,  # None = all users
                'n_eval_items': 100,
                'early_stopping_patience': 10,
                'log_level': 1
            },
            'data': {
                'cache_dir': None,
                'item_feature': 'metadata',
                'neg_option': 'neg-ignore',
                'rating_threshold': 3.5,
                'train_ratio_pos': 0.8,
            },
            'mlflow': {
                'experiment_name': 'movielens_pipeline',
                'tracking_uri': 'sqlite:///mlflow.db'
            },
            'output': {
                'output_dir': './output'
            },
            'grid_search': {
                'enabled': False,
                'param_grid': {
                    'model.n_latent': [32, 64, 128],
                    'optimizer.lr': [0.001, 0.01],
                    'training.loss_function': ['bpr_loss', 'hinge_loss']
                }
            }
        }

    @staticmethod
    def save_config(config: Dict, output_path: str):
        """Save configuration to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(
                config, f, default_flow_style=False, sort_keys=False
            )
        print(f"Configuration saved to: {output_path}")
