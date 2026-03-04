"""Streamlined recommendation system with hybrid MF."""

import sys
from typing import Callable, Dict, List, Optional, Tuple, Union

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.optim
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .interactions import UserItemData
from .mf import MatrixFactorization


class RecommendationSystem:
    """Recommendation system using hybrid MF with BPR optimization."""

    def __init__(
        self,
        uidata: UserItemData,
        model: MatrixFactorization,
        optimizer: Callable[..., torch.optim.Optimizer],
        loss: Callable,
        device: Union[torch.device, str],
    ):
        self.device = (
            torch.device(device) if isinstance(device, str) else device
        )

        # Interaction matrices
        self.Rpos_train_csr = uidata.Rpos_train.tocsr()
        self.Rpos_test_csr = uidata.Rpos_test.tocsr()
        self.Rneg_csr = uidata.Rneg.tocsr()
        self.Fu_csr = uidata.Fu.tocsr()
        self.Fi_csr = uidata.Fi.tocsr()
        self.Rpos_all_csr = uidata.Rpos.tocsr()

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

        # Model, optimizer, loss
        self.model = model.to(self.device)
        self.optimizer = optimizer(self.model.parameters())
        self.loss = loss

        # Training: all users with train interactions
        # Eval: only users present in both splits
        self.train_users = (
            np.diff(self.Rpos_train_csr.indptr).nonzero()[0]
        )
        users_test = np.diff(self.Rpos_test_csr.indptr).nonzero()[0]
        self.eval_users = np.intersect1d(self.train_users, users_test)
        print(
            f'MatrixFactorization | device={self.device}'
            f' | train_users={len(self.train_users)}'
            f' | eval_users={len(self.eval_users)}'
        )

        # Pre-build per-user eval pools (static across evals)
        self._eval_pools = self._build_eval_pools()

    @staticmethod
    def _check_one_hot(csr: csr_matrix) -> bool:
        """Check if every row has exactly one nonzero (one-hot)."""
        return bool(np.all(np.diff(csr.indptr) == 1))

    def _score(
        self,
        users: np.ndarray,
        items: np.ndarray,
    ) -> torch.Tensor:
        """Score pairs; uses embedding lookup or dense matmul."""
        # User embeddings: one-hot → feat_map lookup, else sparse
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

        # Item embeddings + bias: one-hot → lookup, else dense mm
        if self._Fi_is_onehot:
            feat_i = torch.from_numpy(
                self._Fi_feat_map[items]  # type: ignore[index]
            ).long()
            i_emb = self.model.item_latent(feat_i)
            i_bias = self.model.item_biases(feat_i).squeeze(-1)
        else:
            i_t = torch.from_numpy(items).long()
            fi = self.Fi_dense[i_t]  # type: ignore[index]
            i_emb = fi @ self.model.item_latent.weight
            i_bias = (
                fi @ self.model.item_biases.weight
            ).squeeze(-1)

        return (u_emb * i_emb).sum(-1) + i_bias

    def _build_eval_pools(
        self,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Pre-compute (test_pos, excl) per eval user."""
        pools: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for uid in self.eval_users:
            ps = self.Rpos_test_csr.indices[
                self.Rpos_test_csr.indptr[uid]:
                self.Rpos_test_csr.indptr[uid + 1]
            ]
            if not len(ps):
                continue
            excl = self.Rpos_all_csr.indices[
                self.Rpos_all_csr.indptr[uid]:
                self.Rpos_all_csr.indptr[uid + 1]
            ].copy()
            pools[uid] = (ps, excl)
        return pools

    def _compute_metrics(
        self,
        interactions: List[Tuple],
        top_k: int,
    ) -> Dict[str, float]:
        """Compute AUC, NDCG@K, Precision@K, Recall@K."""
        if not interactions:
            return {
                'auc': 0.0, f'ndcg_{top_k}': 0.0,
                f'precision_{top_k}': 0.0,
                f'recall_{top_k}': 0.0,
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
            dcg = (top / log2_denom).sum()
            idcg = (1.0 / log2_denom[:min(n_pos, top_k)]).sum()
            ndcg.append(float(dcg / idcg) if idcg > 0 else 0.0)

        return {
            'auc': float(np.nanmean(auc_scores)),
            f'ndcg_{top_k}': float(np.nanmean(ndcg)),
            f'precision_{top_k}': float(np.nanmean(prec)),
            f'recall_{top_k}': float(np.nanmean(rec)),
        }

    def _sample_pairs(
        self,
        users: List[int],
        pos_csr: csr_matrix,
        neg_csr: csr_matrix,
        excl_csr: csr_matrix,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample BPR pairs (user, pos_item, neg_item) from CSR."""
        pos_ip, pos_ix = pos_csr.indptr, pos_csr.indices
        neg_ip, neg_ix = neg_csr.indptr, neg_csr.indices
        excl_ip, excl_ix = excl_csr.indptr, excl_csr.indices
        n_items = pos_csr.shape[1]

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
            out_u[count] = u
            out_p[count] = ps[np.random.randint(len(ps))]
            if len(ns):
                out_n[count] = ns[np.random.randint(len(ns))]
            else:
                excl = set(excl_ix[excl_ip[u]:excl_ip[u + 1]])
                neg_i = np.random.randint(n_items)
                while neg_i in excl:
                    neg_i = np.random.randint(n_items)
                out_n[count] = neg_i
            count += 1

        return out_u[:count], out_p[:count], out_n[:count]

    def _train(self, batch_users: np.ndarray) -> float:
        """Train one batch, returns BPR loss."""
        self.model.train()
        users, pos_items, neg_items = self._sample_pairs(
            batch_users,
            self.Rpos_train_csr,
            self.Rneg_csr,
            self.Rpos_train_csr,
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
        n_eval_items: Optional[int],
        top_k: int,
        max_users: Optional[int],
    ) -> Dict[str, float]:
        """Evaluate AUC, NDCG@K, Precision@K, Recall@K on test set."""
        self.model.eval()
        users = (
            self.eval_users.tolist()
            if max_users is None or max_users >= len(self.eval_users)
            else np.random.choice(
                self.eval_users, size=max_users, replace=False
            ).tolist()
        )

        with torch.no_grad():
            # Sample negatives via rejection; O(n_pos/n_items) ≈ 0
            n_items = self.Fi_csr.shape[0]
            users_set = set(users)
            interactions: List[Tuple] = []
            for uid, (ps, excl) in self._eval_pools.items():
                if uid not in users_set:
                    continue
                if n_eval_items is None:
                    # All eligible items (user opts in to large alloc)
                    mask = np.ones(n_items, dtype=bool)
                    mask[excl] = False
                    negs_arr = np.where(mask)[0]
                else:
                    # Rejection sampling; rejection rate ≈ 0
                    excl_set = set(excl.tolist())
                    negs: List[int] = []
                    while len(negs) < n_eval_items:
                        candidates = np.random.randint(
                            0, n_items, size=n_eval_items * 2
                        )
                        negs.extend(
                            int(c) for c in candidates
                            if c not in excl_set
                        )
                    negs_arr = np.array(negs[:n_eval_items])
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
            metrics['bpr_loss'] = self.loss(r_ui, r_uj).item()

        self.model.train()
        return metrics

    def fit(
        self,
        n_iter: int,
        n_eval_items: Optional[int] = 100,
        batch_size: int = 1000,
        eval_every: int = 5,
        n_eval_users: Optional[int] = None,
        early_stopping_patience: int = 10,
        top_k: int = 10,
    ) -> None:
        """Train the model using BPR optimization."""
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

        mlflow.log_params({
            'n_iter': n_iter,
            'n_eval_items': n_eval_items,
            'batch_size': batch_size,
            'eval_every': eval_every,
            'n_eval_users': n_eval_users,
            'early_stopping_patience': early_stopping_patience,
        })

        best_ndcg, best_epoch, patience_counter = 0.0, 0, 0

        epoch_looper = tqdm(
            range(1, n_iter + 1),
            total=n_iter,
            file=sys.stdout,
            desc='HybBPR',
            ncols=70,
            unit='ep',
        )

        for epoch in epoch_looper:
            avg_loss = sum(
                self._train(batch[0].numpy()) for batch in dataloader
            ) / len(dataloader)
            epoch_looper.set_postfix({'loss': f'{avg_loss:.4f}'})
            mlflow.log_metric('train_bpr_loss', avg_loss, step=epoch)

            if epoch % eval_every == 0:
                metrics = self.evaluate(
                    n_eval_items=n_eval_items,
                    top_k=top_k,
                    max_users=n_eval_users,
                )
                mlflow.log_metrics(metrics, step=epoch)

                current_ndcg = metrics.get(f'ndcg_{top_k}', 0)
                if current_ndcg > best_ndcg:
                    best_ndcg, best_epoch, patience_counter = (
                        current_ndcg, epoch, 0
                    )
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

                epoch_looper.write(
                    f'E{epoch}:'
                    f' AUC {metrics.get("auc", 0):.3f}'
                    f' | NDCG@{top_k}'
                    f' {metrics.get(f"ndcg_{top_k}", 0):.3f}'
                    f' | P@{top_k}'
                    f' {metrics.get(f"precision_{top_k}", 0):.3f}'
                    f' | R@{top_k}'
                    f' {metrics.get(f"recall_{top_k}", 0):.3f}'
                    f' | Loss {metrics.get("bpr_loss", 0):.3f}'
                )

        if best_epoch > 0:
            self.save_model(name=f'best_model_epoch_{best_epoch}')

    def save_model(self, name: str = 'model', tqdm_obj=None) -> None:
        """Save model to MLflow."""
        try:
            mlflow.pytorch.log_model(
                pytorch_model=self.model, name=name
            )
            msg = f'Logged model to MLflow: {name}'
            (tqdm_obj.write if tqdm_obj else print)(msg)
        except Exception as e:
            print(f'ERROR: Failed to save model: {e}')
            raise
