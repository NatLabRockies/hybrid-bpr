"""Public API exports for hybrid-bpr package."""

# Core data structures
from .interactions import UserItemData

# Loss functions
from .losses import LossFn

# Models
from .mf import MatrixFactorization
from .recommender import RecommendationSystem

# Data loading
from .movielens import (
    load_movielens, MovieLensDownloader,
    load_movielens_ui, FEATURE_DATASETS
)
from .zazzle import load_zazzle as load_zazzle

# Pipeline
from .pipeline import TrainingPipeline

# Utilities
from .utils import init_hero_mlflow
