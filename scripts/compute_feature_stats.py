from tqdm import tqdm
import torch

def compute_feature_stats(train_loader):
    """
    Compute means and standard deviations per feature (ignoring item_id, which is at index 0)
    over all valid (non-padded) timesteps in the training data.
    
    Assumes that each batch from train_loader is a tuple: (X, y, lengths),
    where X has shape (batch_size, max_seq_len, num_features) and the first feature (column 0) 
    is item_id.
    """
    sum_features = None
    sum_sq_features = None
    total_count = 0

    for batch in tqdm(train_loader):
        X, y, lengths = batch  # X: (B, T, F)
        batch_size, max_seq_len, num_features = X.shape
        device = X.device

        mask = torch.arange(max_seq_len, device=device).expand(batch_size, max_seq_len) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(2)  # shape: (B, T, 1)

        X_features = X[:, :, 1:]  # shape: (B, T, F-1)

        # Expand the mask to cover all features (except item_id).
        mask = mask.expand(-1, -1, X_features.size(2))  # shape: (B, T, F-1)

        # Select only valid values.
        X_valid = X_features[mask].view(-1, X_features.size(2))  # shape: (total_valid, F-1)

        # Initialize accumulators if this is the first batch.
        if sum_features is None:
            sum_features = X_valid.sum(dim=0)
            sum_sq_features = (X_valid ** 2).sum(dim=0)
        else:
            sum_features += X_valid.sum(dim=0)
            sum_sq_features += (X_valid ** 2).sum(dim=0)

        total_count += X_valid.size(0)

    # Compute the mean and standard deviation for each feature.
    means = sum_features / total_count
    variances = (sum_sq_features / total_count) - (means ** 2)
    stds = torch.sqrt(variances)

    return means, stds
