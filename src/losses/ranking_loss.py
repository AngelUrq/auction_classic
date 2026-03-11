import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RankingLoss(nn.Module):
    """DeepHit pairwise ranking loss for single-risk survival analysis.

    For each valid pair (i, j) where duration_i < duration_j and i is uncensored,
    penalizes the model when the predicted CDF for i at time T_i is not sufficiently
    higher than the CDF for j at time T_i.

    loss = rank_mat_ij * exp(-(F_i(T_i) - F_j(T_i)) / sigma)
    """

    def __init__(self, sigma: float = 0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, logits: Tensor, listing_durations: Tensor, events: Tensor) -> Tensor:
        """Compute the DeepHit ranking loss.

        Args:
            logits: Raw logits from survival head (N, n_time_bins).
            listing_durations: Observed duration bin index, 0 to n_time_bins-1 (N,).
            events: 1.0 = sold (uncensored), 0.0 = expired (censored) (N,).

        Returns:
            Scalar mean ranking loss.
        """
        pmf = F.softmax(logits, dim=-1)                         # (N, T)
        durations = listing_durations.long()

        rank_mat = self._build_rank_matrix(durations, events)   # (N, N)
        r = self._compute_cdf_difference(pmf, durations)        # (N, N)

        loss = rank_mat * torch.exp((-r / self.sigma).clamp(max=10))  # (N, N)
        loss = loss.mean(dim=1)                                 # (N,)
        return loss.mean()

    def _build_rank_matrix(self, durations: Tensor, events: Tensor) -> Tensor:
        """Build the pairwise comparison matrix.

        rank_mat[i, j] = 1 if duration_i < duration_j and i had an event (uncensored).

        Args:
            durations: Observed duration indices (N,).
            events: 1.0 = sold (uncensored), 0.0 = expired (censored) (N,).

        Returns:
            Binary matrix of valid pairs (N, N).
        """
        n = durations.shape[0]
        duration_row = durations.view(-1, 1).expand(n, n)
        duration_col = durations.view(1, -1).expand(n, n)
        event_row = events.view(-1, 1).expand(n, n)

        rank_mat = (duration_row < duration_col).float() * event_row
        return rank_mat

    def _compute_cdf_difference(self, pmf: Tensor, durations: Tensor) -> Tensor:
        """Compute the CDF difference matrix R.

        R_ij = F_i(T_i) - F_j(T_i), where F is the predicted CDF.

        Args:
            pmf: Predicted probability mass function (N, T).
            durations: Observed duration indices (N,).

        Returns:
            CDF difference matrix (N, N).
        """
        n = pmf.shape[0]
        y = torch.zeros_like(pmf).scatter(1, durations.view(-1, 1), 1.0)  # (N, T) one-hot

        cdf = pmf.cumsum(dim=1)                     # (N, T)
        cdf_at_times = cdf.matmul(y.T)              # (N, N): entry (i, j) = F_i(T_j)

        diag = cdf_at_times.diag().view(1, -1)      # (1, N): F_i(T_i)
        ones = torch.ones((n, 1), device=pmf.device)
        r = ones.matmul(diag) - cdf_at_times        # (N, N): R_ij = F_i(T_i) - F_j(T_i)
        return r.T
