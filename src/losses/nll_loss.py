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
        # Log-probabilities over all time bins for each sample
        log_pmf = F.log_softmax(logits, dim=-1)                              # (N, T)

        duration_indices = listing_durations.unsqueeze(1)                    # (N, 1)

        # Uncensored (sold): -log P(T = t)
        # The model should assign high probability exactly at the observed bin.
        nll_uncensored = -log_pmf.gather(1, duration_indices).squeeze(1)     # (N,)

        # Censored (expired): -log S(t) = -log P(T >= t)
        # The auction was still alive at time t; we don't know when it sold after that.
        # S(t) = P(T >= t) = sum of pmf for bins t, t+1, ..., T-1.
        # Computed as logsumexp in log-space to avoid the numerically unstable
        # 1 - CDF(t) formulation, which produces exploding gradients as CDF(t) -> 1.
        # Using >= (not >) ensures the last bin (t = T-1) always has a non-empty tail.
        n_bins = logits.shape[-1]
        bin_indices = torch.arange(n_bins, device=logits.device)             # (T,)
        tail_mask = bin_indices.unsqueeze(0) >= duration_indices             # (N, T)
        masked_log_pmf = log_pmf.masked_fill(~tail_mask, -float('inf'))     # (N, T)
        log_survival = torch.logsumexp(masked_log_pmf, dim=-1)              # (N,)
        nll_censored = -log_survival

        # Each sample contributes exactly one term:
        #   sold    (events=1): point log-likelihood at observed bin
        #   expired (events=0): log-survival at censoring time
        loss = events * nll_uncensored + (1.0 - events) * nll_censored
        return loss.mean()
