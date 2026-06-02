import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HazardSurvivalLoss(nn.Module):
    """Discrete-time logistic hazard loss (nnet-survival; Gensheimer & Narasimhan 2019).

    Each time bin has an independent hazard h_j = sigmoid(logit_j), the conditional
    probability the event occurs in bin j given survival to j. Unlike the DeepHit
    softmax parametrization, bins do not compete for a shared probability budget, so
    censored auctions cannot pull all the mass into a tail/sink bin — they only
    contribute "survived" (target 0) terms for the bins they passed through.

    For a subject observed at bin k with event indicator e (1 = sold, 0 = expired):

        loss = -[ sum_{j < k} log(1 - h_j)  +  e * log(h_k) + (1 - e) * log(1 - h_k) ]

    i.e. binary cross-entropy over bins 0..k with target 1 only at bin k when the
    event occurred (sold).
    """

    def forward(self, logits: Tensor, listing_durations: Tensor, events: Tensor) -> Tensor:
        """Compute the discrete-time hazard NLL.

        Args:
            logits: Per-bin hazard logits (N, n_time_bins).
            listing_durations: Observed duration bin index, 0 to n_time_bins-1 (N,).
            events: 1.0 = sold (uncensored), 0.0 = expired (censored) (N,).

        Returns:
            Scalar mean loss (summed over at-risk bins, averaged over subjects).
        """
        n_bins = logits.shape[-1]
        bin_indices = torch.arange(n_bins, device=logits.device)             # (T,)
        observed_bin = listing_durations.long().unsqueeze(1)                 # (N, 1)

        # Bins 0..k are "at risk": the subject was observed alive entering each of them.
        at_risk_mask = (bin_indices.unsqueeze(0) <= observed_bin).float()    # (N, T)

        # Target is 1 only at the observed bin and only when the event was observed (sold).
        # Every earlier at-risk bin keeps target 0 ("survived this bin").
        targets = torch.zeros_like(logits)                                   # (N, T)
        targets.scatter_(1, observed_bin, events.unsqueeze(1).to(logits.dtype))  # (N, T)

        per_bin_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # (N, T)
        auction_loss = (per_bin_loss * at_risk_mask).sum(dim=-1)        # (N,)
        return auction_loss.mean()
