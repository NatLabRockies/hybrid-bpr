#!/usr/bin/env python3
"""MovieLens dataset downloader with preprocessing support."""

import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import urllib3
import pandas as pd
import requests

from .interactions import UserItemData


class MovieLensDownloader:
    """Download and load MovieLens datasets with tag support."""

    BASE_URL = "https://files.grouplens.org/datasets/movielens/"

    DATASETS = {
        'ml-100k': {
            'url': 'ml-100k.zip',
            'extract_dir': 'ml-100k',
            'size': '5MB',
            'ratings': 100000,
            'has_tags': False,
            'file_mappings': {
                'ratings': 'u.data',
                'movies': 'u.item',
                'users': 'u.user'
            }
        },
        'ml-1m': {
            'url': 'ml-1m.zip',
            'extract_dir': 'ml-1m',
            'size': '6MB',
            'ratings': 1000000,
            'has_tags': False,
            'file_mappings': {
                'ratings': 'ratings.dat',
                'movies': 'movies.dat',
                'users': 'users.dat'
            }
        },
        'ml-10m': {
            'url': 'ml-10m.zip',
            'extract_dir': 'ml-10M100K',
            'size': '63MB',
            'ratings': 10000000,
            'has_tags': True,
            'file_mappings': {
                'ratings': 'ratings.dat',
                'movies': 'movies.dat',
                'tags': 'tags.dat'
            }
        },
        'ml-20m': {
            'url': 'ml-20m.zip',
            'extract_dir': 'ml-20m',
            'size': '190MB',
            'ratings': 20000000,
            'has_tags': True,
            'file_mappings': {
                'ratings': 'ratings.csv',
                'movies': 'movies.csv',
                'tags': 'tags.csv',
                'links': 'links.csv',
                'genome_scores': 'genome-scores.csv',
                'genome_tags': 'genome-tags.csv'
            }
        },
        'ml-25m': {
            'url': 'ml-25m.zip',
            'extract_dir': 'ml-25m',
            'size': '250MB',
            'ratings': 25000000,
            'has_tags': True,
            'file_mappings': {
                'ratings': 'ratings.csv',
                'movies': 'movies.csv',
                'tags': 'tags.csv',
                'links': 'links.csv',
                'genome_scores': 'genome-scores.csv',
                'genome_tags': 'genome-tags.csv'
            }
        }
    }

    def __init__(
        self,
        cache_dir: Optional[str] = None
    ) -> None:
        """Initialize downloader with cache directory."""
        # Setup cache directory
        cache_path = Path(cache_dir or tempfile.gettempdir())
        self.cache_dir = cache_path / "movielens"
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def download_dataset(
        self,
        dataset: str = 'ml-20m',
        force_download: bool = False
    ) -> Path:
        """Download and extract dataset, returns path."""
        # Validate dataset
        if dataset not in self.DATASETS:
            available = list(self.DATASETS.keys())
            msg = f"Dataset {dataset} not supported. Use: {available}"
            raise ValueError(msg)

        # Setup paths
        dataset_info = self.DATASETS[dataset]
        zip_path = self.cache_dir / f"{dataset}.zip"
        extract_path = self.cache_dir / dataset_info['extract_dir']

        # Check if already downloaded and extracted
        if extract_path.exists() and not force_download:
            return extract_path

        # Download dataset (SSL verification disabled for grouplens)
        url = f"{self.BASE_URL}{dataset_info['url']}"
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(url, stream=True, verify=False)
        response.raise_for_status()

        # Save zip file
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.cache_dir)

        return extract_path

    def _load_file(
        self,
        file_path: Path,
        dataset: str
    ) -> Optional[pd.DataFrame]:
        """Load file based on extension and dataset type."""
        try:
            # Handle CSV files
            if file_path.suffix == '.csv':
                return pd.read_csv(file_path)

            # Handle .dat files
            elif file_path.suffix == '.dat':
                if dataset == 'ml-1m':
                    if 'ratings' in file_path.name:
                        return pd.read_csv(
                            file_path, sep='::',
                            names=[
                                'userId', 'movieId', 'rating',
                                'timestamp'
                            ],
                            engine='python'
                        )
                    elif 'movies' in file_path.name:
                        return pd.read_csv(
                            file_path, sep='::',
                            names=['movieId', 'title', 'genres'],
                            engine='python', encoding='latin1'
                        )
                    elif 'users' in file_path.name:
                        return pd.read_csv(
                            file_path, sep='::',
                            names=[
                                'userId', 'gender', 'age',
                                'occupation', 'zipCode'
                            ],
                            engine='python'
                        )

                elif dataset == 'ml-10m':
                    if 'ratings' in file_path.name:
                        return pd.read_csv(
                            file_path, sep='::',
                            names=[
                                'userId', 'movieId', 'rating',
                                'timestamp'
                            ],
                            engine='python'
                        )
                    elif 'movies' in file_path.name:
                        return pd.read_csv(
                            file_path, sep='::',
                            names=['movieId', 'title', 'genres'],
                            engine='python', encoding='latin1'
                        )
                    elif 'tags' in file_path.name:
                        return pd.read_csv(
                            file_path, sep='::',
                            names=[
                                'userId', 'movieId', 'tag',
                                'timestamp'
                            ],
                            engine='python'
                        )

            # Handle ml-100k .data files
            elif dataset == 'ml-100k' and file_path.suffix == '.data':
                return pd.read_csv(
                    file_path, sep='\t',
                    names=['userId', 'movieId', 'rating', 'timestamp']
                )

            # Handle ml-100k .item files
            elif dataset == 'ml-100k' and file_path.suffix == '.item':
                base_cols = [
                    'movieId', 'title', 'release_date',
                    'video_release_date', 'imdb_url'
                ]
                genre_cols = [f'genre_{i}' for i in range(19)]
                return pd.read_csv(
                    file_path, sep='|',
                    names=base_cols + genre_cols,
                    encoding='latin1'
                )

            # Handle ml-100k .user files
            elif dataset == 'ml-100k' and file_path.suffix == '.user':
                return pd.read_csv(
                    file_path, sep='|',
                    names=[
                        'userId', 'age', 'gender', 'occupation',
                        'zipCode'
                    ]
                )

            else:
                return None

        except Exception:
            return None

    def load_dataset_with_tags(
        self,
        dataset: str = 'ml-20m'
    ) -> Dict[str, pd.DataFrame]:
        """Load complete MovieLens dataset with all files."""
        # Download dataset if needed
        dataset_path = self.download_dataset(dataset)
        data = {}

        # Get dataset-specific file mappings
        dataset_info = self.DATASETS[dataset]
        file_mappings = dataset_info['file_mappings']

        # Load each file if it exists
        for key, filename in file_mappings.items():
            file_path = dataset_path / filename
            if file_path.exists():
                df = self._load_file(file_path, dataset)
                if df is not None:
                    data[key] = df

        return data

    def get_user_tags(
        self,
        tags_df: pd.DataFrame,
        user_id: int
    ) -> pd.DataFrame:
        """Get all tags applied by a specific user."""
        return tags_df[tags_df['userId'] == user_id]

    def get_movie_tags(
        self,
        tags_df: pd.DataFrame,
        movie_id: int
    ) -> pd.DataFrame:
        """Get all tags applied to a specific movie."""
        return tags_df[tags_df['movieId'] == movie_id]

    def print_dataset_summary(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> None:
        """Print summary statistics for all loaded DataFrames."""
        # Define columns to check and their display names
        columns_to_check = {
            'userId': 'Unique Users',
            'movieId': 'Unique Movies',
            'tag': 'Unique Tags',
            'tagId': 'Unique Tag IDs',
            'genres': 'Unique Genres',
            'title': 'Unique Titles'
        }

        for df_name, df in data.items():
            print(f"{df_name.upper()}" + "-" * 20)
            print(f"Total rows: {len(df):,}")

            # Check each potential column
            found_columns = []
            for col, display_name in columns_to_check.items():
                if col in df.columns:
                    unique_count = df[col].nunique()
                    print(f"{display_name}: {unique_count:,}")
                    found_columns.append(col)

            # Show sample columns if no standard ones found
            if not found_columns:
                print(f"Columns: {list(df.columns)}")
                if len(df.columns) <= 5:
                    for col in df.columns[:3]:
                        dtype = df[col].dtype
                        if (dtype == 'object' or
                                dtype.name.startswith('int')):
                            unique_count = df[col].nunique()
                            print(f"Unique {col}: {unique_count:,}")

    def load_and_preprocess(
        self,
        dataset: str = 'ml-100k'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and preprocess MovieLens data for BPR training."""
        # Download and load dataset
        data = self.load_dataset_with_tags(dataset)

        # Get ratings DataFrame
        rdf = data['ratings'].copy()
        rdf.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

        # ml-100k uses movie genres as features
        movies = data['movies'].copy()

        # Extract genre columns (last 19 columns are genre indicators)
        genre_cols = [f'genre_{i}' for i in range(19)]
        genre_data = movies[['movieId'] + genre_cols].copy()

        # Create item features from genres
        item_features = []
        for _, row in genre_data.iterrows():
            movie_id = row['movieId']
            for i, genre_col in enumerate(genre_cols):
                if row[genre_col] == 1:
                    item_features.append(
                        {'MovieID': movie_id, 'TagID': i}
                    )

        tdf = pd.DataFrame(item_features)

        # Keep only movies with features
        rdf = rdf[rdf.MovieID.isin(tdf.MovieID.unique())].copy()

        print(
            f"Loaded {len(rdf)} ratings for {rdf.UserID.nunique()} "
            f"users and {rdf.MovieID.nunique()} movies"
        )
        print(f"Loaded {len(tdf)} genre features")

        return rdf, tdf


# Public API function for easy data loading
def load_movielens(
    dataset: str = 'ml-100k',
    cache_dir: Optional[str] = None,
    preprocess: bool = True
) -> Dict[str, pd.DataFrame]:
    """Load MovieLens dataset; preprocesses ml-100k genre features."""
    downloader = MovieLensDownloader(cache_dir=cache_dir)

    # Preprocess ml-100k with genre features if requested
    if dataset == 'ml-100k' and preprocess:
        rdf, tdf = downloader.load_and_preprocess('ml-100k')
        return {'ratings': rdf, 'features': tdf}

    # Return raw data for all other cases
    return downloader.load_dataset_with_tags(dataset)


# Datasets with item features available for training
FEATURE_DATASETS = ['ml-100k', 'ml-10m', 'ml-20m', 'ml-25m']


def _preprocess_ml10m(raw: dict) -> dict:
    """Normalize ml-10m text tags → integer feature IDs + IDF weights."""
    # Rename rating columns to match standard format
    rdf = raw['ratings'].copy()
    rdf.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

    # Factorize free-text tags to unique integer IDs per movie
    tdf = raw['tags'][['movieId', 'tag']].drop_duplicates().copy()
    tdf['TagID'] = pd.factorize(tdf['tag'])[0]
    tdf = tdf.rename(columns={'movieId': 'MovieID'})[
        ['MovieID', 'TagID']
    ]

    # Compute IDF weights: rare tags are more discriminative
    n_movies = tdf.MovieID.nunique()
    doc_freq = tdf.groupby('TagID')['MovieID'].nunique()
    idf = np.log(n_movies / doc_freq).rename('idf')
    tdf = tdf.join(idf, on='TagID')
    tdf = tdf.rename(columns={'idf': 'relevance'})

    # Keep only movies with tag features; this also ensures negatives
    # from rdf are restricted to tagged movies only (Fix 1)
    rdf = rdf[rdf.MovieID.isin(tdf.MovieID.unique())].copy()
    return {'ratings': rdf, 'features': tdf}


def _preprocess_mlgenome(
    raw: dict,
    relevance_threshold: float = 0.8
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


def load_movielens_ui(
    dataset: str,
    rating_threshold: float,
    item_feature: str,
    name: Optional[str] = None,
    genome_relevance_threshold: float = 0.8
) -> UserItemData:
    """Load MovieLens dataset and return a UserItemData object."""
    # Load and normalize dataset
    if dataset == 'ml-100k':
        data = load_movielens(dataset='ml-100k', preprocess=True)
    elif dataset == 'ml-10m':
        raw = load_movielens(dataset='ml-10m', preprocess=False)
        data = _preprocess_ml10m(raw)
    else:
        raw = load_movielens(dataset=dataset, preprocess=False)
        data = _preprocess_mlgenome(
            raw, relevance_threshold=genome_relevance_threshold
        )

    rdf = data['ratings']
    tdf = data['features']

    # Use relevance as feature weight if present, else uniform
    weights = (
        tdf['relevance'].values
        if 'relevance' in tdf.columns
        else None
    )

    # Initialize UserItemData
    ui = UserItemData(name=name or dataset)

    # Add positive interactions (ratings at or above threshold)
    pos_mask = rdf.Rating >= rating_threshold
    ui.add_positive_interactions(
        user_ids=rdf.UserID[pos_mask].values,
        item_ids=rdf.MovieID[pos_mask].values
    )

    # Always add explicit negative interactions (below threshold)
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


def main() -> None:
    """Demonstrate usage of simple loading function."""
    print("=" * 60)
    print("Example 1: Load ml-100k (preprocessed)")
    print("=" * 60)
    data = load_movielens(dataset='ml-100k', preprocess=True)
    print(f"Keys: {list(data.keys())}")
    print(f"Ratings shape: {data['ratings'].shape}")
    print(f"Features shape: {data['features'].shape}")
    print("\nRatings sample:")
    print(data['ratings'].head(3))

    print("\n" + "=" * 60)
    print("Example 2: Load ml-100k (raw)")
    print("=" * 60)
    data = load_movielens(dataset='ml-100k', preprocess=False)
    print(f"Keys: {list(data.keys())}")
    for key, df in data.items():
        print(f"  {key}: {df.shape}")

    print("\n" + "=" * 60)
    print("Example 3: Load ml-1m")
    print("=" * 60)
    data = load_movielens(dataset='ml-1m')
    print(f"Keys: {list(data.keys())}")
    for key, df in data.items():
        print(f"  {key}: {df.shape}")

    print("\n" + "=" * 60)
    print("Example 4: Load ml-20m (with tags)")
    print("=" * 60)
    data = load_movielens(dataset='ml-20m')
    print(f"Keys: {list(data.keys())}")
    for key, df in data.items():
        print(f"  {key}: {df.shape}")


if __name__ == "__main__":
    main()
