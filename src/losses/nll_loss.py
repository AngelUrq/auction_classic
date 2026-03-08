import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class NLLSurvivalLoss(nn.Module):
    """DeepHit negative log-likelihood loss.

    For uncensored (sold) auctions, maximizes the log-probability of the observed
    duration bin. For censored (expired) auctions, maximizes the log-survival
    probability at the observed time.
    """

    def forward(self, logits: Tensor, listing_durations: Tensor, events: Tensor) -> Tensor:
        """Compute NLL survival loss.

        Args:
            logits: Raw logits from survival head (N, n_time_bins).
            listing_durations: Observed duration bin index, 0 to n_time_bins-1 (N,).
            events: 1.0 = sold (uncensored), 0.0 = expired (censored) (N,).

        Returns:
            Scalar mean NLL loss.
        """
        log_pmf = F.log_softmax(logits, dim=-1)
        pmf = F.softmax(logits, dim=-1)
        cdf = pmf.cumsum(dim=-1)

        duration_indices = listing_durations.unsqueeze(1)  # (N, 1)

        nll_uncensored = -log_pmf.gather(1, duration_indices).squeeze(1)  # (N,)

        survival_probability = (1.0 - cdf.gather(1, duration_indices).squeeze(1)).clamp(min=1e-7)
        nll_censored = -torch.log(survival_probability)  # (N,)

        loss = events * nll_uncensored + (1.0 - events) * nll_censored
        return loss.mean()
