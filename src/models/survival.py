import torch
import torch.nn.functional as F
from torch import Tensor


def survival_pmf(logits: Tensor) -> Tensor:
    """Convert per-bin hazard logits to the discrete event PMF.

    Decode counterpart of the discrete-time hazard parametrization used by
    HazardSurvivalLoss: each bin carries an independent conditional hazard
    h_j = sigmoid(logit_j), the probability the event (sale) occurs in bin j given
    survival up to bin j. The unconditional probability the event occurs in bin j is

        p_j = h_j * prod_{i < j} (1 - h_i)

    The mass not covered by any bin, prod_j (1 - h_j), is P(survives beyond horizon)
    — i.e. never sells within the window. So sum_j p_j = 1 - P(survives beyond horizon),
    which plays the role the old DeepHit "sink" bin used to.

    Args:
        logits: Per-bin hazard logits (..., n_time_bins).

    Returns:
        Discrete event PMF of the same shape; sums to <= 1 along the last dim.
    """
    log_hazard = F.logsigmoid(logits)                                    # log h_j
    log_one_minus_hazard = F.logsigmoid(-logits)                         # log(1 - h_j)
    # Exclusive cumulative sum: sum_{i < j} log(1 - h_i) = log S(j), survival to start of j.
    log_survival_to_bin = torch.cumsum(log_one_minus_hazard, dim=-1) - log_one_minus_hazard
    log_pmf = log_hazard + log_survival_to_bin
    return log_pmf.exp()
