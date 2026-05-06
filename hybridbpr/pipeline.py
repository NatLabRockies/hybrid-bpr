"""Training pipeline for PyBPR recommendation system."""

import sqlite3
import yaml
import torch
import itertools
import logging
from functools import partial
from typing import Dict, List, Any, Optional, Callable

import mlflow
from pathos.multiprocessing import ProcessPool, cpu_count

from .recommender import RecommendationSystem, _mlflow_log
from .interactions import UserItemData
from .mf import MatrixFactorization
from .losses import LossFn

# Suppress verbose MLflow/alembic migration logs
logging.getLogger("alembic").setLevel(logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)


def _configure_sqlite(tracking_uri: str) -> None:
    """Enable WAL mode on SQLite DB to reduce write contention."""
    if not tracking_uri.startswith('sqlite:///'):
        return
    db_path = tracking_uri[len('sqlite:///'):]
    with sqlite3.connect(db_path) as conn:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA busy_timeout=60000')


def _set_experiment_safe(
    mlflow_module: Any, name: str
) -> None:
    """Set MLflow experiment, restoring it if deleted."""
    client = mlflow_module.MlflowClient()
    exp = client.get_experiment_by_name(name)
    if exp is not None and exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)
    mlflow_module.set_experiment(name)


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
            'Adagrad': torch.optim.Adagrad,
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
        num_processes: Optional[int] = None,
        ui_factory: Optional[Callable[[Dict], UserItemData]] = None,
    ) -> List[str]:
        """Run training; sweep=True runs full grid search."""
        # Use custom mlflow if provided, otherwise use default
        mlflow_module = custom_mlflow if custom_mlflow is not None else mlflow

        # Set MLflow tracking and experiment only if custom mlflow not supplied
        if custom_mlflow is None:
            mlflow_module.set_tracking_uri(self.cfg['mlflow.tracking_uri'])
        _set_experiment_safe(
            mlflow_module, self.cfg['mlflow.experiment_name']
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
                num_processes=num_processes,
                custom_mlflow=custom_mlflow,
                ui_factory=ui_factory,
            )
            print(
                f"\nSweep completed: {len(results)} experiments"
            )
            return results
        else:
            print("\nTraining single model...")
            with mlflow_module.start_run() as run:
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
            run_name = "run"

        # Log all config parameters to MLflow
        _mlflow_log(mlflow_module.log_params, self.cfg)

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

        # Split data into train/test according to split_mode
        split_mode = self.cfg.get('data.split_mode', 'warm')
        print(
            f"Starting training: {run_name}"
            f" | split_mode={split_mode}",
            flush=True,
        )
        if split_mode == 'cold':
            ui.split_train_test_cold(
                cold_item_ratio=self.cfg.get(
                    'data.cold_item_ratio', 0.2
                ),
                random_state=self.cfg.get(
                    'data.random_state', None
                ),
            )
        elif split_mode == 'warm':
            ui.split_train_test(
                train_ratio=self.cfg.get(
                    'data.warm_train_ratio', 0.8
                ),
                train_ratio_neg=self.cfg.get(
                    'data.warm_train_ratio', 0.8
                ),
                random_state=self.cfg.get(
                    'data.random_state', None
                ),
            )
        else:
            raise ValueError(
                f"Unknown split_mode '{split_mode}'."
                " Use 'warm' or 'cold'."
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
            use_negs_for_training=self.cfg.get(
                'data.use_negs_for_training', False
            ),
        )

        # Train the model
        recommender.fit(
            n_iter=self.cfg['training.n_iter'],
            batch_size=self.cfg['training.batch_size'],
            eval_every=self.cfg['training.eval_every'],
            n_eval_users=self.cfg.get('training.n_eval_users'),
            top_k=self.cfg.get('training.top_k', 10),
            neg_ratio=self.cfg.get('training.neg_ratio', 1.0),
            early_stopping_patience=self.cfg[
                'training.early_stopping_patience'
            ],
            custom_mlflow=custom_mlflow,
        )

        print(f"Finished training: {run_name}", flush=True)
        return recommender

    def run_grid_search(
        self,
        ui: UserItemData,
        param_grid: Dict[str, List],
        mlflow_experiment_name: Optional[str] = None,
        num_processes: Optional[int] = None,
        custom_mlflow: Optional[Any] = None,
        ui_factory: Optional[
            Callable[[Dict], UserItemData]
        ] = None,
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
            _set_experiment_safe(mlflow_module, mlflow_experiment_name)

        # Enable WAL mode for SQLite to reduce write contention
        if custom_mlflow is None:
            _configure_sqlite(self.cfg['mlflow.tracking_uri'])

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

        # Create partial function with fixed ui and optional factory
        run_single = partial(
            self._run_single_experiment,
            ui=ui,
            custom_mlflow=custom_mlflow,
            ui_factory=ui_factory,
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
        custom_mlflow: Optional[Any] = None,
        ui_factory: Optional[
            Callable[[Dict], UserItemData]
        ] = None,
    ) -> str:
        """Run one experiment; returns SUCCESS/FAILED status string."""
        # Use custom mlflow if provided, otherwise use default
        mlflow_module = (
            custom_mlflow if custom_mlflow is not None else mlflow
        )

        # Set tracking URI in subprocess (multiprocessing resets state)
        if custom_mlflow is None:
            mlflow_module.set_tracking_uri(
                self.cfg['mlflow.tracking_uri']
            )
        _set_experiment_safe(
            mlflow_module, self.cfg['mlflow.experiment_name']
        )

        run_name = "unknown"
        try:
            # Create copy of config for this experiment
            experiment_config = self.cfg_raw.copy()

            # Generate run name from parameters (no dataset prefix)
            run_name_parts: List[str] = []

            # Update experiment config with param grid values
            for param_name, param_value in params.items():
                if '.' in param_name:
                    section, key = param_name.split('.', 1)
                    if section not in experiment_config:
                        experiment_config[section] = {}
                    experiment_config[section][key] = param_value

                    # Add param value to run name
                    if key == 'loss_function' and callable(
                        param_value
                    ):
                        run_name_parts.append(
                            param_value.__name__
                        )
                    else:
                        run_name_parts.append(param_value)

            run_name = '_'.join(str(p) for p in run_name_parts)

            # Create new pipeline with experiment config
            experiment_pipeline = TrainingPipeline(
                config=experiment_config
            )

            # Rebuild ui if factory provided and data params changed
            experiment_ui = ui
            if ui_factory is not None:
                data_params = {
                    k: v for k, v in params.items()
                    if k.startswith('data.')
                }
                if data_params:
                    experiment_ui = ui_factory(
                        experiment_pipeline.cfg
                    )

            # Start MLflow run for this experiment
            with mlflow_module.start_run(run_name=run_name):
                # Train model with MLflow run
                experiment_pipeline.train(
                    ui=experiment_ui, run_name=run_name,
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
                'early_stopping_patience': 10,
                'log_level': 1
            },
            'data': {
                'cache_dir': None,
                'item_feature': 'metadata',
                'rating_threshold': 3.0,
                'warm_train_ratio': 0.8,
                'use_negs_for_training': False,
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
