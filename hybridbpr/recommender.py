"""Streamlined recommendation system with hybrid MF."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import contextlib
import time
import random

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.optim
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .interactions import UserItemData
from .mf import MatrixFactorization

_LOCK_PHRASES = ('database is locked', 'TEMPORARILY_UNAVAILABLE')


def _mlflow_log(fn: Callable, *args, **kwargs) -> None:
    """Call fn(*args, **kwargs) with retry on SQLite lock errors."""
    for attempt in range(12):
        try:
            fn(*args, **kwargs)
            return
        except Exception as e:
            msg = str(e)
            if any(p in msg for p in _LOCK_PHRASES):
                delay = min(
                    0.5 * (2 ** attempt)
                    + random.uniform(0, 0.5),
                    30.0,
                )
                time.sleep(delay)
            else:
                raise
    fn(*args, **kwargs)  # final attempt; let it raise


class RecommendationSystem:
    """Recommendation system using hybrid MF with BPR optimization."""

    def __init__(
        self,
        uidata: UserItemData,
        model: MatrixFactorization,
        optimizer: Callable[..., torch.optim.Optimizer],
        loss: Callable,
        device: Union[torch.device, str],
        use_negs_for_training: bool = False,
    ):
        self.device = (
            torch.device(device) if isinstance(device, str) else device
        )
        self.use_negs_for_training = use_negs_for_training

        # Interaction matrices
        self.Rpos_train_csr = uidata.Rpos_train.tocsr()
        self.Rpos_test_csr = uidata.Rpos_test.tocsr()
        self.Rneg_train_csr = uidata.Rneg_train.tocsr()
        self.Rneg_test_csr = uidata.Rneg_test.tocsr()
        self.Fu_csr = uidata.Fu.tocsr()
        self.Fi_csr = uidata.Fi.tocsr()
        self.Rpos_all_csr = uidata.Rpos.tocsr()
        # Empty neg CSR used when use_negs_for_training=False
        self._empty_neg_csr = csr_matrix(
            self.Rpos_train_csr.shape, dtype=np.float32
        )

        # One-hot check: one nnz per row → embedding lookup via
        # feat_map, avoiding scipy→torch sparse conversion each step
        self._Fu_is_onehot = self._check_one_hot(self.Fu_csr)
        self._Fi_is_onehot = self._check_one_hot(self.Fi_csr)

        # Pre-compute feature index maps for one-hot matrices
        self._Fu_feat_map: Optional[np.ndarray] = (
            self.Fu_csr.indices.copy()
            if self._Fu_is_onehot else None
        )
        self._Fi_feat_map: Optional[np.ndarray] = (
            self.Fi_csr.indices.copy()
            if self._Fi_is_onehot else None
        )

        # Pre-densify Fi if not one-hot: one-time cost, then
        # each step does Fi_dense[items] (fast index) + torch.mm
        if not self._Fi_is_onehot:
            fi_mb = (
                self.Fi_csr.shape[0]
                * self.Fi_csr.shape[1] * 4 / 1e6
            )
            print(
                f'Pre-densifying Fi {self.Fi_csr.shape}'
                f' ({fi_mb:.0f} MB)'
            )
            self.Fi_dense: Optional[torch.Tensor] = (
                torch.from_numpy(
                    np.asarray(
                        self.Fi_csr.todense(), dtype=np.float32
                    )
                ).to(self.device)
            )
        else:
            self.Fi_dense = None
        print(
            f'Fu one-hot={self._Fu_is_onehot}'
            f' | Fi one-hot={self._Fi_is_onehot}'
        )

        # Item embedding cache (populated during eval, None otherwise)
        self._item_emb_cache: Optional[torch.Tensor] = None
        self._item_bias_cache: Optional[torch.Tensor] = None

        # Model, optimizer, loss
        self.model = model.to(self.device)
        self.optimizer = optimizer(self.model.parameters())
        self.loss = loss

        # Training: all users with train interactions
        # Eval: users with both train and test interactions,
        # sorted by descending total cold item interactions so
        # n_eval_users always picks the most-interacted cold users
        self.train_users = (
            np.diff(self.Rpos_train_csr.indptr).nonzero()[0]
        )
        # Require at least one cold pos interaction; negs are
        # random-filled from cold items in evaluate() when absent
        users_cold_pos = (
            np.diff(self.Rpos_test_csr.indptr).nonzero()[0]
        )
        eval_users = np.intersect1d(
            self.train_users, users_cold_pos
        )
        cold_counts = (
            np.diff(self.Rpos_test_csr.indptr)
            + np.diff(self.Rneg_test_csr.indptr)
        )
        self.eval_users = eval_users[
            np.argsort(cold_counts[eval_users])[::-1]
        ]
        print(
            f'MatrixFactorization | device={self.device}'
            f' | train_users={len(self.train_users)}'
            f' | eval_users={len(self.eval_users)}'
        )

        # Pre-build per-user eval pools (static across evals)
        self._eval_pools = self._build_eval_pools()

        # Training neg pool: exclude cold (test-only) items so random
        # negative sampling never leaks cold items into training
        n_items = self.Rpos_train_csr.shape[1]
        cold_items = (
            set(self.Rpos_test_csr.indices.tolist())
            | set(self.Rneg_test_csr.indices.tolist())
        ) - (
            set(self.Rpos_train_csr.indices.tolist())
            | set(self.Rneg_train_csr.indices.tolist())
        )
        if cold_items:
            self._train_neg_pool: np.ndarray = np.array(
                [i for i in range(n_items) if i not in cold_items],
                dtype=np.int64,
            )
            print(
                f'Train neg pool: {len(self._train_neg_pool)}'
                f' warm items'
                f' ({len(cold_items)} cold items excluded)'
            )
        else:
            self._train_neg_pool = np.arange(
                n_items, dtype=np.int64
            )

    @staticmethod
    def _check_one_hot(csr: csr_matrix) -> bool:
        """Check if every row has exactly one nonzero (one-hot)."""
        return bool(np.all(np.diff(csr.indptr) == 1))

    @contextlib.contextmanager
    def _item_cache_ctx(self):
        """Pre-compute item embs/biases for non-one-hot Fi eval."""
        if not self._Fi_is_onehot:
            # Fused matmul: single pass over Fi_dense for emb+bias
            # (n_items, n_tags) x (n_tags, n_latent+1) then split
            fused_w = torch.cat(
                [
                    self.model.item_latent.weight,
                    self.model.item_biases.weight,
                ],
                dim=1,
            )
            fused = self.Fi_dense @ fused_w  # type: ignore[operator]
            self._item_emb_cache = fused[:, :-1]
            # Pre-squeezed: (n_items,) avoids squeeze in _score hot path
            self._item_bias_cache = fused[:, -1]
        try:
            yield
        finally:
            self._item_emb_cache = None
            self._item_bias_cache = None

    def _score(
        self,
        users: np.ndarray,
        items: np.ndarray,
    ) -> torch.Tensor:
        """Score pairs; uses embedding lookup or dense matmul."""
        # User embeddings: one-hot → feat_map lookup, else sparse mm
        if self._Fu_is_onehot:
            feat_u = torch.from_numpy(
                self._Fu_feat_map[users]  # type: ignore[index]
            ).long()
            u_emb = self.model.user_latent(feat_u)
        else:
            u_emb = self.model._safe_sparse_mm(
                self.Fu_csr[users],
                self.model.user_latent.weight,
            )

        # Item embeddings + bias: one-hot → lookup, else index cache
        if self._Fi_is_onehot:
            feat_i = torch.from_numpy(
                self._Fi_feat_map[items]  # type: ignore[index]
            ).long()
            i_emb = self.model.item_latent(feat_i)
            i_bias = self.model.item_biases(feat_i).squeeze(-1)
        elif self._item_emb_cache is not None:
            # Cache hit: O(batch) index into pre-computed embs
            i_t = torch.from_numpy(items).long()
            i_emb = self._item_emb_cache[i_t]
            i_bias = self._item_bias_cache[i_t]
        else:
            # Training path: no cache, compute per-batch
            i_t = torch.from_numpy(items).long()
            fi = self.Fi_dense[i_t]  # type: ignore[index]
            i_emb = fi @ self.model.item_latent.weight
            i_bias = (
                fi @ self.model.item_biases.weight
            ).squeeze(-1)

        return (u_emb * i_emb).sum(-1) + i_bias

    def _build_eval_pools(
        self,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Pre-compute (test_pos, neg_test, excl) per eval user."""
        # Cold mode: test items are disjoint from train items;
        # store cold item set to restrict random fill in evaluate()
        # so warm-item scores don't contaminate cold-start eval
        train_item_set = set(
            self.Rpos_train_csr.indices.tolist()
        )
        test_item_set = set(
            self.Rpos_test_csr.indices.tolist()
        )
        cold_only = sorted(test_item_set - train_item_set)
        self._eval_cold_items: Optional[np.ndarray] = (
            np.array(cold_only, dtype=np.int64)
            if cold_only else None
        )
        print(
            'Eval mode: '
            + (
                f'cold | {len(cold_only)} cold items'
                if self._eval_cold_items is not None
                else 'warm'
            )
        )

        pools: Dict[
            int, Tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        for uid in self.eval_users:
            ps = self.Rpos_test_csr.indices[
                self.Rpos_test_csr.indptr[uid]:
                self.Rpos_test_csr.indptr[uid + 1]
            ]
            if not len(ps):
                continue
            # Explicit test negatives for this user (may be empty;
            # evaluate() random-fills from cold items when absent)
            neg_test = self.Rneg_test_csr.indices[
                self.Rneg_test_csr.indptr[uid]:
                self.Rneg_test_csr.indptr[uid + 1]
            ].copy()
            # Exclude all known positives from random sampling
            excl = self.Rpos_all_csr.indices[
                self.Rpos_all_csr.indptr[uid]:
                self.Rpos_all_csr.indptr[uid + 1]
            ].copy()
            pools[uid] = (ps, neg_test, excl)
        return pools

    def _compute_metrics(
        self,
        interactions: List[Tuple],
        top_k: int,
    ) -> Dict[str, float]:
        """Compute AUC, NDCG@K, Precision@K, Recall@K."""
        if not interactions:
            return {
                'AUC-Test': 0.0,
                f'NDCG:{top_k}-Test': 0.0,
                f'Precision:{top_k}-Test': 0.0,
                f'Recall:{top_k}-Test': 0.0,
            }

        # Build flat index arrays for a single batched forward pass
        user_idx: List[np.ndarray] = []
        item_idx: List[np.ndarray] = []
        n_pos_list: List[int] = []
        n_total_list: List[int] = []
        for uid, pos_items, neg_items in interactions:
            all_items = np.concatenate([pos_items, neg_items])
            n_total = len(all_items)
            user_idx.append(
                np.full(n_total, uid, dtype=np.int64)
            )
            item_idx.append(all_items)
            n_pos_list.append(len(pos_items))
            n_total_list.append(n_total)

        all_u = np.concatenate(user_idx)
        all_i = np.concatenate(item_idx)

        # Single batched prediction, chunked to bound memory
        _CHUNK = 100_000
        preds_all = np.empty(len(all_u), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(all_u), _CHUNK):
                e = min(s + _CHUNK, len(all_u))
                preds_all[s:e] = self._score(
                    all_u[s:e], all_i[s:e]
                ).numpy()

        # Compute per-user metrics from flat predictions
        auc_scores: List[float] = []
        ndcg: List[float] = []
        prec: List[float] = []
        rec: List[float] = []
        log2_denom = np.log2(np.arange(2, top_k + 2))
        offset = 0
        for n_pos, n_total in zip(n_pos_list, n_total_list):
            preds = preds_all[offset:offset + n_total]
            offset += n_total
            y = np.zeros(n_total, dtype=np.int8)
            y[:n_pos] = 1
            auc_scores.append(roc_auc_score(y, preds))
            top = y[np.argsort(preds)[::-1][:top_k]].astype(
                np.float32
            )
            hits = int(top.sum())
            prec.append(hits / top_k)
            rec.append(hits / n_pos if n_pos else 0.0)
            dcg = (top / log2_denom[:len(top)]).sum()
            idcg = (1.0 / log2_denom[:min(n_pos, top_k)]).sum()
            ndcg.append(float(dcg / idcg) if idcg > 0 else 0.0)

        return {
            'AUC-Test': float(np.nanmean(auc_scores)),
            f'NDCG:{top_k}-Test': float(np.nanmean(ndcg)),
            f'Precision:{top_k}-Test': float(np.nanmean(prec)),
            f'Recall:{top_k}-Test': float(np.nanmean(rec)),
        }

    def _sample_pairs(
        self,
        users: List[int],
        pos_csr: csr_matrix,
        neg_csr: csr_matrix,
        excl_csr: csr_matrix,
        neg_pool: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample BPR pairs (user, pos_item, neg_item) from CSR."""
        pos_ip, pos_ix = pos_csr.indptr, pos_csr.indices
        neg_ip, neg_ix = neg_csr.indptr, neg_csr.indices
        excl_ip, excl_ix = excl_csr.indptr, excl_csr.indices
        n_items = pos_csr.shape[1]
        pool_len = (
            len(neg_pool) if neg_pool is not None else n_items
        )

        # Pre-allocate; slice to actual count at the end
        n = len(users)
        out_u = np.empty(n, dtype=np.int64)
        out_p = np.empty(n, dtype=np.int64)
        out_n = np.empty(n, dtype=np.int64)
        count = 0

        for u in users:
            ps = pos_ix[pos_ip[u]:pos_ip[u + 1]]
            if not len(ps):
                continue
            ns = neg_ix[neg_ip[u]:neg_ip[u + 1]]
            excl = set(excl_ix[excl_ip[u]:excl_ip[u + 1]])
            out_u[count] = u
            out_p[count] = ps[np.random.randint(len(ps))]
            if len(ns):
                # Filter explicit negs that overlap with positives
                valid_ns = [n for n in ns if n not in excl]
                if valid_ns:
                    out_n[count] = valid_ns[
                        np.random.randint(len(valid_ns))
                    ]
                else:
                    # All explicit negs were positives; random fallback
                    out_n[count] = self._rand_neg(
                        excl, neg_pool, pool_len
                    )
            else:
                out_n[count] = self._rand_neg(
                    excl, neg_pool, pool_len
                )
            count += 1

        return out_u[:count], out_p[:count], out_n[:count]

    @staticmethod
    def _rand_neg(
        excl: set,
        neg_pool: Optional[np.ndarray],
        pool_len: int,
    ) -> int:
        """Sample a random negative not in excl from pool."""
        while True:
            idx = np.random.randint(pool_len)
            item = int(neg_pool[idx]) if neg_pool is not None else idx
            if item not in excl:
                return item

    def _train(self, batch_users: np.ndarray) -> float:
        """Train one batch, returns BPR loss."""
        self.model.train()
        neg_src = (
            self.Rneg_train_csr
            if self.use_negs_for_training
            else self._empty_neg_csr
        )
        users, pos_items, neg_items = self._sample_pairs(
            batch_users,
            self.Rpos_train_csr,
            neg_src,
            self.Rpos_train_csr,
            neg_pool=self._train_neg_pool,
        )
        r_ui = self._score(users, pos_items)
        r_uj = self._score(users, neg_items)
        self.optimizer.zero_grad()
        loss = self.loss(r_ui, r_uj)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(
        self,
        top_k: int,
        max_users: Optional[int],
        neg_ratio: float = 1.0,
    ) -> Dict[str, float]:
        """Evaluate AUC, NDCG@K, Precision@K, Recall@K on test set."""
        self.model.eval()
        # eval_users is pre-sorted by descending cold interactions;
        # take top max_users for deterministic, high-signal eval
        users = (
            self.eval_users.tolist()
            if max_users is None or max_users >= len(self.eval_users)
            else self.eval_users[:max_users].tolist()
        )

        with torch.no_grad(), self._item_cache_ctx():
            n_items = self.Fi_csr.shape[0]
            users_set = set(users)
            interactions: List[Tuple] = []
            for uid, (ps, neg_test, excl) in (
                self._eval_pools.items()
            ):
                if uid not in users_set:
                    continue

                # Always use all explicit negs; random-fill only the
                # shortfall up to neg_ratio * n_pos. In cold mode,
                # restrict random fill to cold items only.
                n_neg = max(1, round(len(ps) * neg_ratio))
                n_explicit = len(neg_test)
                n_random = max(0, n_neg - n_explicit)
                if n_random == 0:
                    negs_arr = neg_test
                else:
                    excl_set = (
                        set(excl.tolist()) | set(neg_test.tolist())
                    )
                    # Cold mode: sample only from cold items
                    pool = (
                        self._eval_cold_items
                        if self._eval_cold_items is not None
                        else np.arange(n_items, dtype=np.int64)
                    )
                    rand_negs: List[int] = []
                    while len(rand_negs) < n_random:
                        candidates = pool[
                            np.random.randint(
                                0, len(pool), size=n_random * 2
                            )
                        ]
                        rand_negs.extend(
                            int(c) for c in candidates
                            if c not in excl_set
                        )
                    random_arr = np.array(rand_negs[:n_random])
                    negs_arr = (
                        np.concatenate([neg_test, random_arr])
                        if n_explicit else random_arr
                    )
                interactions.append((uid, ps, negs_arr))

            metrics = self._compute_metrics(interactions, top_k)

            # BPR test loss (random negatives, no explicit test negs)
            empty_neg = csr_matrix(
                self.Rpos_test_csr.shape, dtype=np.float32
            )
            users_t, pos_t, neg_t = self._sample_pairs(
                users,
                self.Rpos_test_csr,
                empty_neg,
                self.Rpos_all_csr,
            )
            r_ui = self._score(users_t, pos_t)
            r_uj = self._score(users_t, neg_t)
            metrics['BPR_Loss-Test'] = self.loss(r_ui, r_uj).item()

        self.model.train()
        return metrics

    def fit(
        self,
        n_iter: int,
        batch_size: int = 1000,
        eval_every: int = 5,
        n_eval_users: Optional[int] = None,
        early_stopping_patience: int = 10,
        top_k: int = 10,
        neg_ratio: float = 1.0,
        custom_mlflow: Optional[Any] = None,
    ) -> None:
        """Train the model using BPR optimization."""
        # Use custom mlflow if provided (e.g. Hero backend)
        mlflow_mod = custom_mlflow if custom_mlflow is not None else mlflow

        from pathos.helpers import mp
        num_workers = (
            0 if mp.current_process().name != 'MainProcess' else 2
        )

        dataloader = DataLoader(
            TensorDataset(torch.LongTensor(self.train_users)),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        print(
            f'# minibatches={len(dataloader):,}'
            f' | eval_every={eval_every}'
        )

        _mlflow_log(mlflow_mod.log_params, {
            'n_iter': n_iter,
            'batch_size': batch_size,
            'eval_every': eval_every,
            'n_eval_users': n_eval_users,
            'early_stopping_patience': early_stopping_patience,
            'neg_ratio': neg_ratio,
        })

        best_ndcg, best_epoch, patience_counter = 0.0, 0, 0
        w = len(str(n_iter))

        for epoch in range(1, n_iter + 1):
            avg_loss = sum(
                self._train(batch[0].numpy()) for batch in dataloader
            ) / len(dataloader)
            _mlflow_log(
                mlflow_mod.log_metric,
                'BPR_Loss-Train', avg_loss, step=epoch,
            )

            # Inline progress bar
            pct = epoch / n_iter
            filled = int(20 * pct)
            bar = '#' * filled + '-' * (20 - filled)
            print(
                f'\rEpoch {epoch:{w}}/{n_iter}'
                f' [{bar}] loss={avg_loss:.4f}',
                end='', flush=True,
            )

            if epoch % eval_every == 0:
                metrics = self.evaluate(
                    top_k=top_k,
                    max_users=n_eval_users,
                    neg_ratio=neg_ratio,
                )
                _mlflow_log(
                    mlflow_mod.log_metrics, metrics, step=epoch
                )

                current_ndcg = metrics.get(f'NDCG:{top_k}-Test', 0)
                if current_ndcg > best_ndcg:
                    best_ndcg, best_epoch, patience_counter = (
                        current_ndcg, epoch, 0
                    )
                else:
                    patience_counter += 1

                # Print eval metrics on new line
                print(
                    f'\nE{epoch}:'
                    f' AUC {metrics.get("AUC-Test", 0):.3f}'
                    f' | NDCG:{top_k}'
                    f' {metrics.get(f"NDCG:{top_k}-Test", 0):.3f}'
                    f' | P:{top_k}'
                    f' {metrics.get(f"Precision:{top_k}-Test", 0):.3f}'
                    f' | R:{top_k}'
                    f' {metrics.get(f"Recall:{top_k}-Test", 0):.3f}'
                    f' | Loss'
                    f' {metrics.get("BPR_Loss-Test", 0):.3f}'
                )

                if patience_counter >= early_stopping_patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

        print()

        if best_epoch > 0:
            self.save_model(
                name=f'best_model_epoch_{best_epoch}',
                custom_mlflow=custom_mlflow,
            )

    def save_model(
        self,
        name: str = 'model',
        custom_mlflow: Optional[Any] = None,
    ) -> None:
        """Save model to MLflow."""
        mlflow_mod = (
            custom_mlflow if custom_mlflow is not None else mlflow
        )
        try:
            mlflow_mod.pytorch.log_model(
                pytorch_model=self.model, name=name
            )
            print(f'Logged model to MLflow: {name}')
        except Exception as e:
            print(f'ERROR: Failed to save model: {e}')
            raise
