#!/usr/bin/env python3
"""User-Item interaction data management."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import dill
import numpy as np
import scipy.sparse as sp
from numpy.random import RandomState


class UserItemData:
    """Manages user-item interaction and feature data."""

    def __init__(
        self,
        name: str,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize UserItemData."""
        self.name = str(name)
        self.dtype = dtype

        # Dimension counters
        self.n_users = 0
        self.n_items = 0
        self.n_user_features = 0
        self.n_item_features = 0

        # Empty sparse matrices
        self._Rpos = sp.coo_matrix((0, 0), dtype=dtype)
        self._Rneg = sp.coo_matrix((0, 0), dtype=dtype)
        self._Fu = sp.coo_matrix((0, 0), dtype=dtype)
        self._Fi = sp.coo_matrix((0, 0), dtype=dtype)

        # Train/test split results (None until split_train_test called)
        self.Rpos_train: Optional[sp.coo_matrix] = None
        self.Rpos_test: Optional[sp.coo_matrix] = None
        self.Rneg_train: Optional[sp.coo_matrix] = None
        self.Rneg_test: Optional[sp.coo_matrix] = None

        # ID <-> index mappings for each entity type
        self._id_to_idx_mappings = {
            'user': (dict(), dict()),
            'item': (dict(), dict()),
            'user_feature': (dict(), dict()),
            'item_feature': (dict(), dict()),
        }

        print(f"Initialized UserItemData '{name}' with dtype {dtype}")

    def _get_index(
        self,
        input_id: int,
        mapping_type: str
    ) -> int:
        """Get or create internal index for a given ID."""
        if mapping_type not in self._id_to_idx_mappings:
            raise ValueError(
                f"Unknown mapping type: {mapping_type}"
            )

        id_to_idx, idx_to_id = (
            self._id_to_idx_mappings[mapping_type]
        )

        if input_id not in id_to_idx:
            idx = len(id_to_idx)
            id_to_idx[input_id] = idx
            idx_to_id[idx] = input_id

            # Update dimension counters
            if mapping_type == 'user':
                self.n_users = max(self.n_users, idx + 1)
            elif mapping_type == 'item':
                self.n_items = max(self.n_items, idx + 1)
            elif mapping_type == 'user_feature':
                self.n_user_features = max(
                    self.n_user_features, idx + 1
                )
            elif mapping_type == 'item_feature':
                self.n_item_features = max(
                    self.n_item_features, idx + 1
                )

        return id_to_idx[input_id]

    def _get_indices(
        self,
        ids: List[int],
        mapping_type: str
    ) -> List[int]:
        """Convert list of IDs to internal indices."""
        return [
            self._get_index(input_id, mapping_type)
            for input_id in ids
        ]

    def get_id(self, idx: int, mapping_type: str) -> int:
        """Get original ID from internal index."""
        if mapping_type not in self._id_to_idx_mappings:
            raise ValueError(
                f"Unknown mapping type: {mapping_type}"
            )

        _, idx_to_id = self._id_to_idx_mappings[mapping_type]
        if idx not in idx_to_id:
            raise ValueError(
                f"{mapping_type.capitalize()} index "
                f"{idx} not found in mapping"
            )
        return idx_to_id[idx]

    def _process_weights(
        self,
        weights: Optional[Union[float, List[float]]],
        length: int
    ) -> np.ndarray:
        """Process and validate interaction/feature weights."""
        # Default to uniform weights
        if weights is None:
            return np.ones(length, dtype=self.dtype)

        # Broadcast scalar to full array
        if np.isscalar(weights):
            if not np.isfinite(weights):
                raise ValueError("Weight must be finite")
            return np.full(length, weights, dtype=self.dtype)

        # Validate array length and finiteness
        if len(weights) != length:
            raise ValueError(
                f"Weight length ({len(weights)}) must match "
                f"input length ({length})"
            )
        weight_array = np.array(weights, dtype=self.dtype)
        if not np.all(np.isfinite(weight_array)):
            raise ValueError("All weights must be finite")
        return weight_array

    def _update_matrix(
        self,
        old_matrix: sp.coo_matrix,
        new_shape: Tuple[int, int],
        new_matrix: Optional[sp.coo_matrix] = None,
    ) -> sp.coo_matrix:
        """Update and resize a sparse matrix."""
        # Validate shapes
        if any(dim < 0 for dim in new_shape):
            raise ValueError(
                "Matrix dimensions must be non-negative"
            )
        if (
            new_matrix is not None
            and new_shape != new_matrix.shape
        ):
            raise ValueError(
                f"Shape mismatch: expected {new_shape}, "
                f"got {new_matrix.shape}"
            )

        # Return new matrix directly if old is empty
        if old_matrix.nnz == 0:
            if new_matrix is not None:
                return new_matrix
            return sp.coo_matrix(new_shape, dtype=self.dtype)

        # Resize existing matrix to new shape if needed
        if (
            old_matrix.shape[0] < new_shape[0]
            or old_matrix.shape[1] < new_shape[1]
        ):
            old_matrix = sp.coo_matrix(
                (old_matrix.data,
                 (old_matrix.row, old_matrix.col)),
                shape=new_shape,
                dtype=self.dtype
            )

        # Merge new entries into existing matrix
        result = old_matrix
        if new_matrix is not None:
            result = old_matrix + new_matrix
            result.eliminate_zeros()
        return result

    def _reshape_all_matrices(self) -> None:
        """Resize all matrices to current dimensions."""
        # Interaction matrices
        self._Rpos = self._update_matrix(
            self._Rpos, (self.n_users, self.n_items)
        )
        self._Rneg = self._update_matrix(
            self._Rneg, (self.n_users, self.n_items)
        )
        # Feature matrices
        self._Fi = self._update_matrix(
            self._Fi, (self.n_items, self.n_item_features)
        )
        self._Fu = self._update_matrix(
            self._Fu, (self.n_users, self.n_user_features)
        )

    def add_interactions(
        self,
        user_ids: List[int],
        item_ids: List[int],
        weights: Optional[Union[float, List[float]]] = None,
        is_positive: bool = True
    ) -> None:
        """Add user-item interactions to the dataset."""
        if len(user_ids) != len(item_ids):
            raise ValueError(
                f"User and item ID lists must have equal length: "
                f"{len(user_ids)} != {len(item_ids)}"
            )
        if len(user_ids) == 0:
            return

        itype = "positive" if is_positive else "negative"
        print(f"Adding {len(user_ids):,} {itype} interactions")

        # Map IDs to internal indices
        user_indices = self._get_indices(user_ids, 'user')
        item_indices = self._get_indices(item_ids, 'item')

        # Build sparse interaction matrix
        values = self._process_weights(weights, len(user_indices))
        interaction_matrix = sp.coo_matrix(
            (values, (user_indices, item_indices)),
            shape=(self.n_users, self.n_items),
            dtype=self.dtype
        )
        interaction_matrix.eliminate_zeros()

        # Update the target matrix (positive or negative)
        target = self._Rpos if is_positive else self._Rneg
        target = self._update_matrix(
            old_matrix=target,
            new_matrix=interaction_matrix,
            new_shape=(self.n_users, self.n_items)
        )
        if is_positive:
            self._Rpos = target
        else:
            self._Rneg = target

        # Resize all matrices to match new dimensions
        self._reshape_all_matrices()

        print(
            f"Added {itype} interactions. "
            f"Dims: {self.n_users}×{self.n_items}"
        )

    def add_positive_interactions(
        self,
        user_ids: List[int],
        item_ids: List[int],
        weights: Optional[Union[float, List[float]]] = None
    ) -> None:
        """Add positive interactions."""
        self.add_interactions(
            user_ids, item_ids, weights, is_positive=True
        )

    def add_negative_interactions(
        self,
        user_ids: List[int],
        item_ids: List[int],
        weights: Optional[Union[float, List[float]]] = None
    ) -> None:
        """Add negative interactions."""
        self.add_interactions(
            user_ids, item_ids, weights, is_positive=False
        )

    def add_user_features(
        self,
        user_ids: List[int],
        feature_ids: List[int],
        feature_weights: Optional[Union[float, List[float]]] = None
    ) -> None:
        """Add user features to the dataset."""
        if len(user_ids) != len(feature_ids):
            raise ValueError(
                f"User and feature ID lists must have equal length: "
                f"{len(user_ids)} != {len(feature_ids)}"
            )
        if len(user_ids) == 0:
            return

        print(f"Adding {len(user_ids):,} user features")

        # Map IDs to indices
        user_indices = self._get_indices(user_ids, 'user')
        feature_indices = self._get_indices(
            feature_ids, 'user_feature'
        )

        # Build and merge feature matrix
        values = self._process_weights(
            feature_weights, len(user_indices)
        )
        feature_matrix = sp.coo_matrix(
            (values, (user_indices, feature_indices)),
            shape=(self.n_users, self.n_user_features),
            dtype=self.dtype
        )
        feature_matrix.eliminate_zeros()
        self._Fu = self._update_matrix(
            old_matrix=self._Fu,
            new_matrix=feature_matrix,
            new_shape=(self.n_users, self.n_user_features)
        )
        self._reshape_all_matrices()

        print(
            f"Added user features: "
            f"{self.n_users}×{self.n_user_features}"
        )

    def add_item_features(
        self,
        item_ids: List[int],
        feature_ids: List[int],
        feature_weights: Optional[Union[float, List[float]]] = None
    ) -> None:
        """Add item features to the dataset."""
        if len(item_ids) != len(feature_ids):
            raise ValueError(
                f"Item and feature ID lists must have equal length: "
                f"{len(item_ids)} != {len(feature_ids)}"
            )
        if len(item_ids) == 0:
            return

        print(f"Adding {len(item_ids):,} item features")

        # Map IDs to indices
        item_indices = self._get_indices(item_ids, 'item')
        feature_indices = self._get_indices(
            feature_ids, 'item_feature'
        )

        # Build and merge feature matrix
        values = self._process_weights(
            feature_weights, len(item_indices)
        )
        feature_matrix = sp.coo_matrix(
            (values, (item_indices, feature_indices)),
            shape=(self.n_items, self.n_item_features),
            dtype=self.dtype
        )
        feature_matrix.eliminate_zeros()
        self._Fi = self._update_matrix(
            old_matrix=self._Fi,
            new_matrix=feature_matrix,
            new_shape=(self.n_items, self.n_item_features)
        )
        self._reshape_all_matrices()

        print(
            f"Added item features: "
            f"{self.n_items}×{self.n_item_features}"
        )

    def validate_dataset(self) -> None:
        """Validate dataset dimensions and content."""
        # Require at least some positive interactions
        if self.Rpos.nnz == 0:
            raise ValueError("No positive interaction data found")

        # Check all matrix shapes match expected dimensions
        matrices = [
            ('Rpos', self.Rpos, (self.n_users, self.n_items)),
            ('Rneg', self.Rneg, (self.n_users, self.n_items)),
            ('Fu', self.Fu, (self.n_users, self.n_user_features)),
            ('Fi', self.Fi, (self.n_items, self.n_item_features)),
        ]
        for mat_name, matrix, expected in matrices:
            if matrix.shape != expected:
                raise ValueError(
                    f"Matrix {mat_name} has shape {matrix.shape}, "
                    f"expected {expected}"
                )

        # Warn if users or items are missing features
        if self._Fu.nnz > 0:
            n_with = len(set(self._Fu.row))
            if n_with < self.n_users:
                print(
                    f"WARNING: {self.n_users - n_with}"
                    f"/{self.n_users} users have no features"
                )
        else:
            print("WARNING: No user features found")

        if self._Fi.nnz > 0:
            n_with = len(set(self._Fi.row))
            if n_with < self.n_items:
                print(
                    f"WARNING: {self.n_items - n_with}"
                    f"/{self.n_items} items have no features"
                )
        else:
            print("WARNING: No item features found")

    def save(self, filepath: Union[str, Path]) -> None:
        """Save instance to file using dill."""
        filepath = Path(filepath)
        try:
            # Create parent directory if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'wb') as f:
                dill.dump(self, f)
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"Saved to {filepath} ({size_mb:.1f} MB)")
        except Exception as e:
            raise IOError(f"Failed to save dataset: {e}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'UserItemData':
        """Load instance from file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            with open(filepath, 'rb') as f:
                instance = dill.load(f)
            if not isinstance(instance, cls):
                raise TypeError(
                    f"Loaded object is not {cls.__name__}, "
                    f"got {type(instance).__name__}"
                )
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(
                f"Loaded dataset from {filepath} "
                f"({size_mb:.1f} MB)"
            )
            return instance

        except Exception as e:
            raise IOError(f"Failed to load dataset: {e}")

    def _get_interaction_stats(
        self,
        matrix: sp.coo_matrix
    ) -> str:
        """Get min/max interaction counts for a matrix."""
        if matrix.nnz == 0:
            return ""

        # Compute per-user and per-item interaction totals
        csr = matrix.tocsr()
        user_counts = np.array(csr.sum(axis=1)).flatten()
        item_counts = np.array(csr.sum(axis=0)).flatten()

        # Filter to entities with at least one interaction
        user_counts = user_counts[user_counts > 0]
        item_counts = item_counts[item_counts > 0]

        stats = ""
        if len(user_counts) > 0:
            stats += (
                f"users: min={int(user_counts.min())}, "
                f"max={int(user_counts.max())} | "
            )
        if len(item_counts) > 0:
            stats += (
                f"items: min={int(item_counts.min())}, "
                f"max={int(item_counts.max())}"
            )
        return stats

    def _mat_str(self, m: sp.coo_matrix) -> str:
        """Compact stats string for a sparse matrix."""
        r, c = m.shape
        density = m.nnz / max(r * c, 1)
        empty_r = r - len(np.unique(m.row)) if m.nnz else r
        empty_c = c - len(np.unique(m.col)) if m.nnz else c
        return (
            f"({r:5}×{c:5}) "
            f"nnz={m.nnz:9,} ({density:5.3%}),"
            f"empty={empty_r:5}/{empty_c:5}"
        )

    def __repr__(self) -> str:
        """Return string representation."""
        istr = f"{self.__class__.__name__}({self.name})\n"
        istr += f"  Fuser: {self._mat_str(self.Fu)}\n"
        istr += f"  Fitem: {self._mat_str(self.Fi)}\n"
        istr += f"  Rpos:  {self._mat_str(self.Rpos)}\n"
        rpos_stats = self._get_interaction_stats(self.Rpos)
        if rpos_stats:
            istr += f"         └─ {rpos_stats}\n"
        istr += f"  Rneg:  {self._mat_str(self.Rneg)}"
        rneg_stats = self._get_interaction_stats(self.Rneg)
        if rneg_stats:
            istr += f"\n         └─ {rneg_stats}"
        return istr

    @property
    def user_ids_in_interactions(self) -> List[int]:
        """Sorted list of user IDs that have interactions."""
        return sorted(
            self._id_to_idx_mappings['user'][0].keys()
        )

    @property
    def item_ids_in_interactions(self) -> List[int]:
        """Sorted list of item IDs that have interactions."""
        return sorted(
            self._id_to_idx_mappings['item'][0].keys()
        )

    @property
    def Rpos(self) -> sp.coo_matrix:
        """Positive interactions matrix."""
        return self._Rpos

    @property
    def Rneg(self) -> sp.coo_matrix:
        """Negative interactions matrix."""
        return self._Rneg

    @property
    def Fu(self) -> sp.coo_matrix:
        """User features matrix."""
        return self._Fu

    @property
    def Fi(self) -> sp.coo_matrix:
        """Item features matrix."""
        return self._Fi

    def _split_matrix(
        self,
        m: sp.coo_matrix,
        train_ratio: float,
        rng: RandomState,
        guarantee_train_entry: bool = True,
    ) -> Tuple[sp.coo_matrix, sp.coo_matrix]:
        """Split a COO matrix into train/test by row-wise ratio."""
        if m.nnz == 0:
            return (
                sp.coo_matrix(m.shape, dtype=self.dtype),
                sp.coo_matrix(m.shape, dtype=self.dtype),
            )

        rows, cols, data = m.row, m.col, m.data

        # Random split
        idx = np.arange(m.nnz)
        rng.shuffle(idx)
        is_train = np.zeros(m.nnz, dtype=bool)
        is_train[idx[:int(m.nnz * train_ratio)]] = True

        # Ensure every user with entries has ≥1 in train
        if guarantee_train_entry:
            for user in np.setdiff1d(
                np.unique(rows), np.unique(rows[is_train])
            ):
                candidates = np.where(
                    (rows == user) & ~is_train
                )[0]
                if len(candidates):
                    is_train[rng.choice(candidates)] = True

        tr = np.where(is_train)[0]
        te = np.where(~is_train)[0]
        m_train = sp.coo_matrix(
            (data[tr], (rows[tr], cols[tr])), shape=m.shape
        )
        m_test = sp.coo_matrix(
            (data[te], (rows[te], cols[te])), shape=m.shape
        )
        return m_train, m_test

    def split_train_test(
        self,
        train_ratio: float = 0.8,
        train_ratio_neg: float = 0.8,
        random_state: Optional[int] = None
    ) -> None:
        """Split pos+neg interactions into train/test sets."""
        for ratio, name in [
            (train_ratio, 'train_ratio'),
            (train_ratio_neg, 'train_ratio_neg'),
        ]:
            if not 0 <= ratio <= 1:
                raise ValueError(
                    f"{name} must be between 0 and 1 inclusive"
                )

        rng = RandomState(random_state)

        # Handle boundary ratios for positives
        m = self._Rpos
        if train_ratio == 0:
            self.Rpos_train = sp.coo_matrix(m.shape, dtype=self.dtype)
            self.Rpos_test = m
        elif train_ratio == 1:
            self.Rpos_train = m
            self.Rpos_test = sp.coo_matrix(m.shape, dtype=self.dtype)
        else:
            self.Rpos_train, self.Rpos_test = self._split_matrix(
                m, train_ratio, rng, guarantee_train_entry=True
            )

        print(
            f"Pos split: {self.Rpos_train.nnz:,} train / "
            f"{self.Rpos_test.nnz:,} test (ratio={train_ratio})"
        )

        # Handle boundary ratios for negatives
        n = self._Rneg
        if n.nnz == 0:
            self.Rneg_train = sp.coo_matrix(n.shape, dtype=self.dtype)
            self.Rneg_test = sp.coo_matrix(n.shape, dtype=self.dtype)
        elif train_ratio_neg == 0:
            self.Rneg_train = sp.coo_matrix(n.shape, dtype=self.dtype)
            self.Rneg_test = n
        elif train_ratio_neg == 1:
            self.Rneg_train = n
            self.Rneg_test = sp.coo_matrix(n.shape, dtype=self.dtype)
        else:
            self.Rneg_train, self.Rneg_test = self._split_matrix(
                n, train_ratio_neg, rng, guarantee_train_entry=True
            )

        print(
            f"Neg split: {self.Rneg_train.nnz:,} train / "
            f"{self.Rneg_test.nnz:,} test"
            f" (ratio={train_ratio_neg})"
        )

    def split_train_test_cold(
        self,
        cold_item_ratio: float = 0.2,
        random_state: Optional[int] = None,
    ) -> None:
        """Cold-start split: rare items→test only, warm→train only.

        Cold items are the least-interacted (by total pos+neg count).
        Ties broken randomly. All cold interactions go to test; all
        warm interactions go to train.
        """
        if not 0 < cold_item_ratio < 1:
            raise ValueError(
                "cold_item_ratio must be strictly between 0 and 1"
            )

        rng = RandomState(random_state)

        # Designate cold items as least-interacted by total count;
        # break ties randomly to avoid systematic bias.
        # Only items with at least one positive interaction are
        # candidates — items without a positive can never contribute
        # to test evaluation (eval requires Rpos_test entries).
        pos_csr = self._Rpos.tocsr()
        neg_csr = self._Rneg.tocsr()
        pos_counts = np.array(pos_csr.sum(axis=0)).flatten()
        neg_counts = np.array(neg_csr.sum(axis=0)).flatten()
        item_counts = pos_counts + neg_counts
        has_pos = np.where(pos_counts > 0)[0]
        noise = rng.uniform(0, 1e-6, size=len(has_pos))
        sorted_by_count = has_pos[
            np.argsort(item_counts[has_pos] + noise, kind='stable')
        ]
        n_cold = max(1, int(len(has_pos) * cold_item_ratio))
        cold_arr = np.sort(sorted_by_count[:n_cold])
        print(
            f"Cold split: {n_cold}/{len(has_pos)} cold items"
            f" ({cold_item_ratio:.0%} of items with positives,"
            f" {self.n_items - len(has_pos)} skipped)"
            f" | max cold interactions:"
            f" {int(item_counts[cold_arr].max())}"
        )

        # Route each interaction to train (warm) or test (cold)
        def _split(
            m: sp.coo_matrix,
        ) -> Tuple[sp.coo_matrix, sp.coo_matrix]:
            if m.nnz == 0:
                empty = sp.coo_matrix(m.shape, dtype=self.dtype)
                return empty, empty
            is_cold = np.isin(m.col, cold_arr)
            tr = np.where(~is_cold)[0]
            te = np.where(is_cold)[0]
            m_train = sp.coo_matrix(
                (m.data[tr], (m.row[tr], m.col[tr])),
                shape=m.shape, dtype=self.dtype,
            )
            m_test = sp.coo_matrix(
                (m.data[te], (m.row[te], m.col[te])),
                shape=m.shape, dtype=self.dtype,
            )
            return m_train, m_test

        # Split positive and negative interactions
        self.Rpos_train, self.Rpos_test = _split(self._Rpos)
        print(
            f"Pos cold split: {self.Rpos_train.nnz:,} train / "
            f"{self.Rpos_test.nnz:,} test"
        )
        self.Rneg_train, self.Rneg_test = _split(self._Rneg)
        print(
            f"Neg cold split: {self.Rneg_train.nnz:,} train / "
            f"{self.Rneg_test.nnz:,} test"
        )
