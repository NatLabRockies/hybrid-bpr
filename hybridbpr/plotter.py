"""MLflow experiment visualization utilities."""

from pathlib import Path
from typing import Optional, List, Dict, Union

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


class MLflowPlotter:
    """Plot metrics from MLflow experiments."""

    def __init__(self, tracking_uri: Union[str, Path] = "mlflow.db"):
        """Initialize plotter with MLflow tracking URI."""
        # Convert path to SQLite URI if needed
        if not str(tracking_uri).startswith(("sqlite://", "http")):
            tracking_uri = f"sqlite:///{tracking_uri}"

        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.MlflowClient(tracking_uri=tracking_uri)

    def get_experiments(self) -> pd.DataFrame:
        """Get all experiments as DataFrame."""
        exps = self.client.search_experiments()
        return pd.DataFrame([
            {
                "experiment_id": e.experiment_id,
                "name": e.name,
                "artifact_location": e.artifact_location,
            }
            for e in exps
        ])

    def get_runs(
        self,
        experiment_name: Optional[str] = None,
        experiment_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Get runs for an experiment as DataFrame."""
        # Get experiment ID from name if needed
        if experiment_id is None and experiment_name is not None:
            exp = self.client.get_experiment_by_name(experiment_name)
            if exp is None:
                raise ValueError(f"Experiment '{experiment_name}' not found")
            experiment_id = exp.experiment_id

        # Search runs
        runs = self.client.search_runs(
            experiment_ids=[experiment_id] if experiment_id else None
        )

        # Build DataFrame
        data = []
        for run in runs:
            row = {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
            }
            row.update(run.data.params)
            row.update(run.data.metrics)
            data.append(row)

        return pd.DataFrame(data)

    def get_run_metrics_history(
        self, run_id: str, metric_keys: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Get metric history for a run as {metric: DataFrame}."""
        # Get all metric keys if not specified
        if metric_keys is None:
            run = self.client.get_run(run_id)
            metric_keys = list(run.data.metrics.keys())

        # Fetch history for each metric
        histories = {}
        for key in metric_keys:
            history = self.client.get_metric_history(run_id, key)
            histories[key] = pd.DataFrame([
                {
                    "step": m.step,
                    "value": m.value,
                    "timestamp": m.timestamp,
                }
                for m in history
            ])

        return histories

    def plot_runs_comparison(
        self,
        experiment_name: str,
        metrics: List[str] = ["train_bpr_loss", "auc"],
        figsize: tuple = (14, 5),
        std_width: float = 1.0,
        show_std: bool = True
    ) -> Figure:
        """Plot loss and eval metrics side by side with std bands."""
        # Get experiment and runs
        exp = self.client.get_experiment_by_name(experiment_name)
        if exp is None:
            raise ValueError(
                f"Experiment '{experiment_name}' not found"
            )
        runs = self.client.search_runs(
            experiment_ids=[exp.experiment_id]
        )
        if len(runs) == 0:
            raise ValueError(
                f"No runs found in experiment '{experiment_name}'"
            )

        # Split into loss vs eval metrics
        loss_metrics = [m for m in metrics if 'loss' in m.lower()]
        eval_metrics = [
            m for m in metrics if 'loss' not in m.lower()
        ]

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        metric_data = {m: {} for m in metrics}

        # Plot loss on left, eval metrics on right
        for metric_name, ax in (
            [(m, axes[0]) for m in loss_metrics]
            + [(m, axes[1]) for m in eval_metrics]
        ):
            for run in runs:
                try:
                    history = self.client.get_metric_history(
                        run.info.run_id, metric_name
                    )
                    if len(history) == 0:
                        continue
                    steps = [m.step for m in history]
                    values = [m.value for m in history]
                    run_name = (
                        run.info.run_name or run.info.run_id[:8]
                    )
                    ax.plot(
                        steps, values,
                        label=f"{run_name} {metric_name}",
                        marker='o', markersize=3, alpha=0.7
                    )
                    for s, v in zip(steps, values):
                        if s not in metric_data[metric_name]:
                            metric_data[metric_name][s] = []
                        metric_data[metric_name][s].append(v)
                except Exception:
                    continue

            # Std bands on non-loss metrics if requested
            if (show_std and 'loss' not in metric_name.lower()
                    and metric_data[metric_name]):
                steps_s = sorted(metric_data[metric_name].keys())
                means = [
                    np.mean(metric_data[metric_name][s])
                    for s in steps_s
                ]
                stds = [
                    np.std(metric_data[metric_name][s])
                    for s in steps_s
                ]
                ax.fill_between(
                    steps_s,
                    [m - std_width * s for m, s in zip(means, stds)],
                    [m + std_width * s for m, s in zip(means, stds)],
                    alpha=0.15, color='gray',
                    label=f'±{std_width}σ'
                )

        # Axis labels and formatting
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("BPR Loss")
        axes[0].set_title("Loss")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].set_title("Eval Metrics")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(
            f"Experiment: {experiment_name}", fontsize=14, y=1.02
        )
        fig.tight_layout()
        return fig

    def plot_single_run(
        self,
        run_id: str,
        figsize: tuple = (14, 5),
        std_width: float = 2.0,
        show_std: bool = True
    ) -> Figure:
        """Plot train loss and eval metrics for a single run."""
        # Get run and all available metrics
        run = self.client.get_run(run_id)
        all_metric_keys = list(run.data.metrics.keys())

        if len(all_metric_keys) == 0:
            raise ValueError(f"No metrics found for run {run_id}")

        # Fetch all metric histories
        histories = self.get_run_metrics_history(
            run_id, all_metric_keys
        )

        # Left: loss metrics; Right: everything else (excl _std keys)
        loss_metrics = [
            k for k in histories if 'loss' in k.lower()
        ]
        eval_metrics = [
            k for k in histories
            if 'loss' not in k.lower() and '_std' not in k.lower()
        ]

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot BPR loss metrics on left subplot
        ax_loss = axes[0]
        for metric_name in loss_metrics:
            df = histories[metric_name]
            if len(df) > 0:
                ax_loss.plot(
                    df["step"], df["value"], marker='o',
                    markersize=4, linewidth=2, label=metric_name
                )
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("BPR Loss")
        ax_loss.set_title("BPR Loss")
        ax_loss.legend(loc="best")
        ax_loss.grid(True, alpha=0.3)

        # Plot eval metrics on right subplot with optional std bands
        ax_eval = axes[1]
        for idx, metric_name in enumerate(eval_metrics):
            df = histories[metric_name]
            if len(df) == 0:
                continue
            color = f'C{idx}'
            ax_eval.plot(
                df["step"], df["value"], marker='o',
                markersize=4, linewidth=2,
                label=metric_name, color=color
            )
            # Add std bands from paired _std metric if present
            df_std = histories.get(f"{metric_name}_std")
            if show_std and df_std is not None and len(df_std) > 0:
                steps = df["step"].values
                means = df["value"].values
                stds = df_std["value"].values
                ax_eval.fill_between(
                    steps,
                    means - std_width * stds,
                    means + std_width * stds,
                    alpha=0.2, color=color,
                    label=f'{metric_name} ±{std_width}σ'
                )

        ax_eval.set_xlabel("Epoch")
        ax_eval.set_ylabel("Score")
        ax_eval.set_title("Eval Metrics (AUC / NDCG / P / R)")
        ax_eval.legend(loc="best")
        ax_eval.grid(True, alpha=0.3)

        run_name = run.info.run_name or run.info.run_id[:8]
        fig.suptitle(f"Run: {run_name}", fontsize=14, y=1.02)
        fig.tight_layout()
        return fig

    def plot_best_runs(
        self,
        experiment_name: str,
        metric: str = "auc",
        n_best: int = 5,
        plot_metrics: List[str] = ["train_bpr_loss", "auc"],
        figsize: tuple = (14, 5),
        std_width: float = 1.0,
        show_std: bool = True
    ) -> Figure:
        """Plot top N runs: BPR loss left, eval metrics right."""
        # Get experiment and top-N runs
        exp = self.client.get_experiment_by_name(experiment_name)
        if exp is None:
            raise ValueError(
                f"Experiment '{experiment_name}' not found"
            )
        runs = self.client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=n_best
        )
        if len(runs) == 0:
            raise ValueError(
                f"No runs found in experiment '{experiment_name}'"
            )

        # Split into loss vs eval metrics
        loss_metrics = [
            m for m in plot_metrics if 'loss' in m.lower()
        ]
        eval_metrics = [
            m for m in plot_metrics if 'loss' not in m.lower()
        ]

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        metric_data = {m: {} for m in plot_metrics}

        # Plot loss on left, eval metrics on right
        for metric_name, ax in (
            [(m, axes[0]) for m in loss_metrics]
            + [(m, axes[1]) for m in eval_metrics]
        ):
            for rank, run in enumerate(runs, 1):
                try:
                    history = self.client.get_metric_history(
                        run.info.run_id, metric_name
                    )
                    if len(history) == 0:
                        continue
                    steps = [m.step for m in history]
                    values = [m.value for m in history]
                    run_name = (
                        run.info.run_name or run.info.run_id[:8]
                    )
                    final_val = run.data.metrics.get(metric, 0)
                    label = (
                        f"#{rank} {run_name} "
                        f"({metric}={final_val:.4f})"
                    )
                    ax.plot(
                        steps, values, label=label, marker='o',
                        markersize=3, alpha=0.7, linewidth=2
                    )
                    for s, v in zip(steps, values):
                        if s not in metric_data[metric_name]:
                            metric_data[metric_name][s] = []
                        metric_data[metric_name][s].append(v)
                except Exception:
                    continue

            # Std bands on eval metrics only
            if (show_std and 'loss' not in metric_name.lower()
                    and metric_data[metric_name]):
                steps_s = sorted(metric_data[metric_name].keys())
                means = [
                    np.mean(metric_data[metric_name][s])
                    for s in steps_s
                ]
                stds = [
                    np.std(metric_data[metric_name][s])
                    for s in steps_s
                ]
                ax.fill_between(
                    steps_s,
                    [m - std_width * s for m, s in zip(means, stds)],
                    [m + std_width * s for m, s in zip(means, stds)],
                    alpha=0.15, color='gray',
                    label=f'±{std_width}σ'
                )

        # Axis formatting
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("BPR Loss")
        axes[0].set_title("BPR Loss")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].set_title("Eval Metrics")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(
            f"Top {n_best} Runs by {metric} - {experiment_name}",
            fontsize=14, y=1.02
        )
        fig.tight_layout()
        return fig

    def summary_table(
        self,
        experiment_name: str,
        metrics: List[str] = ["auc", "train_bpr_loss"],
        params: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Create summary DataFrame of runs sorted by first metric."""
        # Get runs
        runs_df = self.get_runs(experiment_name=experiment_name)

        # Select columns
        cols = ["run_name", "status"]
        if params:
            cols.extend([p for p in params if p in runs_df.columns])
        cols.extend([m for m in metrics if m in runs_df.columns])

        return runs_df[cols].sort_values(
            by=metrics[0] if metrics else "run_name", ascending=False
        )
