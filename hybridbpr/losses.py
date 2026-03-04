"""Pairwise ranking loss functions."""
import torch


class LossFn:
    """Namespace for built-in pairwise ranking loss functions."""

    @staticmethod
    def bpr_loss(
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        """BPR loss (negative log-sigmoid of score difference)."""
        diff = positive_scores - negative_scores
        return -torch.log(torch.sigmoid(sigma * diff)).mean()

    @staticmethod
    def bpr_loss_v2(
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
    ) -> torch.Tensor:
        """BPR loss (separate sigmoid terms)."""
        loss = -torch.log(torch.sigmoid(positive_scores))
        loss += -torch.log(torch.sigmoid(-negative_scores))
        return loss.mean()

    @staticmethod
    def hinge_loss(
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """Hinge (margin ranking) loss."""
        diff = positive_scores - negative_scores
        return torch.nn.functional.relu(margin - diff).mean()

    @staticmethod
    def warp_loss(
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
        num_items: int,
    ) -> torch.Tensor:
        """WARP loss (simplified rank-weighted hinge)."""
        diff = positive_scores - negative_scores
        rank = torch.sum(diff < 0).float() + 1
        weight = (
            torch.log(rank + 1)
            / torch.log(torch.tensor(num_items, dtype=torch.float))
        )
        return (weight * torch.nn.functional.relu(1 - diff)).mean()
