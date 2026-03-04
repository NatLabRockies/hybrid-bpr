"""Shared utilities for hybrid-bpr run scripts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import TrainingPipeline


def init_hero_mlflow(pipeline: 'TrainingPipeline') -> tuple:
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
