#!/usr/bin/env python
"""Hybrid matrix factorization model for recommendation systems."""

from typing import Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


class MatrixFactorization(nn.Module):
    """Matrix Factorization model with user/item feature support."""

    def __init__(
        self,
        n_user_features: int,
        n_item_features: int,
        n_latent: int,
        sparse: bool = False,
        init_std: float = 0.1,
    ) -> None:
        """Initialize hybrid matrix factorization model."""
        super().__init__()
        self.n_latent = n_latent
        self.n_user_features = n_user_features
        self.n_item_features = n_item_features

        # User/item embedding layers
        self.user_latent = nn.Embedding(
            n_user_features, n_latent, sparse=sparse
        )
        self.item_latent = nn.Embedding(
            n_item_features, n_latent, sparse=sparse
        )
        self.item_biases = nn.Embedding(
            n_item_features, 1, sparse=sparse
        )

        # Weight init and FAISS index placeholder
        self._initialize_weights(init_std)
        self._faiss_index = None

    def _initialize_weights(self, init_std: float) -> None:
        """Initialize weights and biases."""
        nn.init.normal_(self.user_latent.weight.data, 0, init_std)
        nn.init.normal_(self.item_latent.weight.data, 0, init_std)
        nn.init.zeros_(self.item_biases.weight.data)

    def _scipy_to_torch_sparse(
        self,
        scipy_matrix: sp.spmatrix,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Convert scipy sparse matrix to PyTorch sparse tensor."""
        coo = scipy_matrix.tocoo()
        indices = torch.from_numpy(
            np.vstack((coo.row, coo.col)).astype(np.int64)
        )
        values = torch.from_numpy(coo.data.astype(np.float32))
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, coo.shape, dtype=torch.float32
        )
        if device is not None:
            sparse_tensor = sparse_tensor.to(device)
        return sparse_tensor.coalesce()

    def _safe_sparse_mm(
        self,
        features: Union[torch.Tensor, sp.spmatrix],
        dense_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse/dense matrix multiply with device handling."""
        if sp.issparse(features):
            features = self._scipy_to_torch_sparse(
                features, device=dense_tensor.device
            )
        if isinstance(features, torch.Tensor):
            if features.device != dense_tensor.device:
                features = features.to(dense_tensor.device)
            if features.is_sparse:
                return torch.sparse.mm(features, dense_tensor)
            return torch.mm(features, dense_tensor)
        raise TypeError(f"Unsupported features type: {type(features)}")

    def forward(
        self,
        user_features: Union[torch.Tensor, sp.spmatrix],
        item_features: Union[torch.Tensor, sp.spmatrix],
    ) -> torch.Tensor:
        """Predict scores for user-item pairs (batch_size,)."""
        # Compute latent vectors
        user_latent = self._safe_sparse_mm(
            user_features, self.user_latent.weight
        )
        item_latent = self._safe_sparse_mm(
            item_features, self.item_latent.weight
        )

        # Compute predictions and add item bias
        predictions = torch.sum(user_latent * item_latent, dim=-1)
        item_bias = self._safe_sparse_mm(
            item_features, self.item_biases.weight
        )
        return predictions + item_bias.squeeze(-1)

    def predict(
        self,
        user_features: Union[torch.Tensor, sp.spmatrix],
        item_features: Union[torch.Tensor, sp.spmatrix],
        item_batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Predict rating matrix (n_users, n_items)."""
        with torch.no_grad():
            n_items = (
                item_features.shape[0] if sp.issparse(item_features)
                else item_features.size(0)
            )
            if item_batch_size is not None and n_items > item_batch_size:
                all_preds = []
                for i in range(0, n_items, item_batch_size):
                    batch_end = min(i + item_batch_size, n_items)
                    batch_items = item_features[i:batch_end, :]
                    all_preds.append(
                        self._predict_unbatched(user_features, batch_items)
                    )
                return torch.cat(all_preds, dim=1)
            return self._predict_unbatched(user_features, item_features)

    def _predict_unbatched(
        self,
        user_features: Union[torch.Tensor, sp.spmatrix],
        item_features: Union[torch.Tensor, sp.spmatrix],
    ) -> torch.Tensor:
        """Unbatched prediction (n_users, n_items)."""
        # Compute latent representations
        user_latent = self._safe_sparse_mm(
            user_features, self.user_latent.weight
        )
        item_latent = self._safe_sparse_mm(
            item_features, self.item_latent.weight
        )
        # Compute score matrix and add item biases
        predictions = user_latent @ item_latent.t()
        item_bias = self._safe_sparse_mm(
            item_features, self.item_biases.weight
        ).squeeze(-1)
        return predictions + item_bias.unsqueeze(0)

    def get_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (user_embeddings, item_embeddings)."""
        return (
            self.user_latent.weight.data.clone(),
            self.item_latent.weight.data.clone(),
        )

    def get_user_embedding(
        self,
        user_features: Union[torch.Tensor, sp.spmatrix],
    ) -> torch.Tensor:
        """Get latent embeddings for given users."""
        return self._safe_sparse_mm(
            user_features, self.user_latent.weight
        )

    def get_item_embedding(
        self,
        item_features: Union[torch.Tensor, sp.spmatrix],
    ) -> torch.Tensor:
        """Get latent embeddings for given items."""
        return self._safe_sparse_mm(
            item_features, self.item_latent.weight
        )

    def build_faiss_index(
        self,
        item_features: Union[torch.Tensor, sp.spmatrix],
        use_gpu: bool = False,
    ) -> None:
        """Build FAISS cosine-similarity index from item embeddings."""
        if not _FAISS_AVAILABLE:
            raise ImportError(
                "faiss required. Install: pip install faiss-cpu"
            )
        # Extract and L2-normalize item embeddings
        item_emb = self.get_item_embedding(item_features)
        emb_np = item_emb.detach().cpu().numpy().astype(np.float32)
        faiss.normalize_L2(emb_np)

        # Create flat inner-product index, optionally on GPU
        index = faiss.IndexFlatIP(emb_np.shape[1])
        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(emb_np)
        self._faiss_index = index

    def top_k(
        self,
        user_features: Union[torch.Tensor, sp.spmatrix],
        k: int = 10,
        exclude_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return top-k (scores, indices) per user via FAISS ANN."""
        if self._faiss_index is None:
            raise RuntimeError(
                "Call build_faiss_index before top_k."
            )
        # Normalize user embeddings for cosine similarity search
        user_emb = self.get_user_embedding(user_features)
        emb_np = user_emb.detach().cpu().numpy().astype(np.float32)
        faiss.normalize_L2(emb_np)

        # Retrieve candidates; full scan when filtering needed
        n_total = self._faiss_index.ntotal
        search_k = (
            n_total if exclude_mask is not None else min(k, n_total)
        )
        scores, indices = self._faiss_index.search(emb_np, search_k)

        # Apply exclusion mask and trim to k
        if exclude_mask is not None:
            scores, indices = self._filter_excluded(
                scores, indices, exclude_mask, k
            )
        return scores[:, :k], indices[:, :k]

    def _filter_excluded(
        self,
        scores: np.ndarray,
        indices: np.ndarray,
        exclude_mask: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove excluded items from FAISS results per user."""
        n_users = scores.shape[0]
        out_s = np.full((n_users, k), -np.inf, dtype=np.float32)
        out_i = np.full((n_users, k), -1, dtype=np.int64)
        for u in range(n_users):
            valid = ~exclude_mask[u, indices[u]]
            vs, vi = scores[u][valid][:k], indices[u][valid][:k]
            out_s[u, :len(vs)] = vs
            out_i[u, :len(vi)] = vi
        return out_s, out_i
