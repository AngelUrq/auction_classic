from tqdm import tqdm
import torch

def compute_feature_stats(train_loader):
    """
    Compute means and standard deviations per feature over all valid (non-padded) 
    timesteps in the training data.
    
    Assumes that each batch from train_loader is a tuple: (X, y, lengths),
    where X has shape (batch_size, max_seq_len, num_features).
    """
    sum_features = None
    sum_sq_features = None
    total_count = 0

    for batch in tqdm(train_loader):
        (auctions, contexts, bonus_lists, modifier_types, modifier_values), y = batch  # X: (B, T, F)
        batch_size, max_seq_len, num_features = auctions.shape
        device = auctions.device

        auction_mask = auctions[:, :, 0] != 0 # padding
        auction_mask = auction_mask.unsqueeze(2)  # shape: (B, T, 1)

        # Expand the mask to cover all features
        mask = auction_mask.expand(-1, -1, auctions.size(2))  # shape: (B, T, F)

        # Select only valid values
        X_valid = auctions[mask].view(-1, auctions.size(2))  # shape: (total_valid, F)

        # Initialize accumulators if this is the first batch
        if sum_features is None:
            sum_features = X_valid.sum(dim=0)
            sum_sq_features = (X_valid ** 2).sum(dim=0)
        else:
            sum_features += X_valid.sum(dim=0)
            sum_sq_features += (X_valid ** 2).sum(dim=0)

        total_count += X_valid.size(0)

    # Compute the mean and standard deviation for each feature
    means = sum_features / total_count
    variances = (sum_sq_features / total_count) - (means ** 2)
    stds = torch.sqrt(variances)

    return means, stds
