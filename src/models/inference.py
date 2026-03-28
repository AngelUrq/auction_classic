import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


MAX_BONUSES = 9
MAX_MODIFIERS = 11


def predict_pmf(model, df_item, feature_stats, max_hours_back=0, max_sequence_length=4096):
    """Run inference for a single item group and return the full PMF over time bins.

    Args:
        model: Trained AuctionTransformer model.
        df_item: DataFrame rows for one item (may include historical context rows).
        feature_stats: Dict with precomputed means/stds for normalization.
        max_hours_back: Maximum historical context window in hours.
        max_sequence_length: Maximum sequence length (crops oldest entries).

    Returns:
        pmf: torch.Tensor of shape (S, n_time_bins) on CPU — probability mass
            over duration bins for each position in the sequence.
        df_item: The sorted and (if needed) cropped DataFrame corresponding to
            the S positions, in old-to-new order.
    """
    df_item = df_item[
        (df_item["snapshot_offset"] >= 0) & (df_item["snapshot_offset"] <= max_hours_back)
    ].sort_values("snapshot_offset", ascending=False)

    if max_sequence_length is not None and len(df_item) > max_sequence_length:
        df_item = df_item.tail(max_sequence_length)

    features_np = np.stack([
        np.log1p(df_item["bid"].to_numpy(dtype=np.float32)),
        np.log1p(df_item["buyout"].to_numpy(dtype=np.float32)),
        df_item["quantity"].to_numpy(dtype=np.float32),
        df_item["time_left"].to_numpy(dtype=np.float32),
        df_item["listing_age"].to_numpy(dtype=np.float32),
    ], axis=1)

    features = torch.tensor(features_np, dtype=torch.float32, device=model.device)
    features = (features - feature_stats["means"][:5].to(model.device)) / (feature_stats["stds"][:5].to(model.device) + 1e-6)

    item_indices = torch.tensor(df_item["item_index"].to_numpy(), dtype=torch.long, device=model.device)
    contexts     = torch.tensor(df_item["context"].to_numpy(), dtype=torch.long, device=model.device)

    bonus_ids_np = np.asarray([
        (row[:MAX_BONUSES] + [0] * (MAX_BONUSES - len(row))) for row in df_item["bonus_ids"]
    ], dtype=np.int64)
    modifier_types_np = np.asarray([
        (row[:MAX_MODIFIERS] + [0] * (MAX_MODIFIERS - len(row))) for row in df_item["modifier_types"]
    ], dtype=np.int64)
    modifier_values_np = np.asarray([
        (row[:MAX_MODIFIERS] + [0.0] * (MAX_MODIFIERS - len(row))) for row in df_item["modifier_values"]
    ], dtype=np.float32)

    bonus_ids       = torch.tensor(bonus_ids_np, dtype=torch.long, device=model.device)
    modifier_types  = torch.tensor(modifier_types_np, dtype=torch.long, device=model.device)
    modifier_values = torch.tensor(modifier_values_np, dtype=torch.float32, device=model.device)
    modifier_values = torch.log1p(modifier_values)
    modifier_values = (modifier_values - feature_stats["modifiers_mean"].to(model.device)) / (feature_stats["modifiers_std"].to(model.device) + 1e-6)

    n_buyout_ranks = model.hparams.n_buyout_ranks
    buyout_rank = torch.tensor(
        np.clip(df_item["buyout_rank"].to_numpy(dtype=np.int64), 0, n_buyout_ranks - 1),
        dtype=torch.long, device=model.device
    )
    hour_of_week    = torch.tensor(df_item["hour_of_week"].to_numpy(), dtype=torch.long, device=model.device)
    snapshot_offset = torch.tensor(
        np.clip(df_item["snapshot_offset"].to_numpy(), 0, max_hours_back),
        dtype=torch.int32, device=model.device
    )

    X = (
        features.unsqueeze(0),
        item_indices.unsqueeze(0),
        contexts.unsqueeze(0),
        bonus_ids.unsqueeze(0),
        modifier_types.unsqueeze(0),
        modifier_values.unsqueeze(0),
        buyout_rank.unsqueeze(0),
        hour_of_week.unsqueeze(0),
        snapshot_offset.unsqueeze(0),
    )

    with torch.no_grad():
        survival_logits = model(X)  # (1, S, n_time_bins)

    pmf = torch.softmax(survival_logits[0], dim=-1).cpu()  # (S, n_time_bins)
    return pmf, df_item


def predict_dataframe(model, df_auctions, prediction_time, feature_stats, max_hours_back=0, max_sequence_length=4096, quick_sale_threshold_hours=12, show_progress=False):
    """Run inference for all items in df_auctions and return predictions at snapshot_offset==0.

    Args:
        model: Trained AuctionTransformer model.
        df_auctions: DataFrame with auction records including historical context.
        prediction_time: Datetime of the prediction snapshot.
        feature_stats: Dict with precomputed means/stds for normalization.
        max_hours_back: Maximum historical context window in hours.
        max_sequence_length: Maximum sequence length per item (crops oldest).
        quick_sale_threshold_hours: Number of hours used to compute sale_probability.
            P(duration < threshold) — proxy for sale probability. Tune this value
            to calibrate the signal; lower = stricter (higher precision).

    Returns:
        df_out: Copy of the filtered DataFrame with added columns:
            prediction_q10, prediction_q50, prediction_q90 (hours),
            expected_duration (hours), sale_probability.
    """
    model.eval()

    df = df_auctions[(df_auctions["snapshot_offset"] >= 0) & (df_auctions["snapshot_offset"] <= max_hours_back)].copy()
    grouped = {idx: g for idx, g in df.groupby("item_index")}

    df_out = df.copy()
    df_out["prediction_q10"]   = np.nan
    df_out["prediction_q50"]   = np.nan
    df_out["prediction_q90"]   = np.nan
    df_out["expected_duration"] = np.nan
    df_out["sale_probability"]  = np.nan

    items_iter = tqdm(grouped.items(), total=len(grouped), desc="Predicting") if show_progress else grouped.items()
    for _, df_item in items_iter:
        pmf, df_item = predict_pmf(model, df_item, feature_stats, max_hours_back, max_sequence_length)
        # pmf: (S, n_time_bins) on CPU

        mask_now = (df_item["snapshot_offset"].to_numpy() == 0)
        if not mask_now.any():
            continue

        cdf = pmf.cumsum(dim=-1)  # (S, n_time_bins)
        n_time_bins = pmf.shape[-1]
        time_bins = torch.arange(n_time_bins, dtype=pmf.dtype)

        q10 = (cdf >= 0.1).float().argmax(dim=-1)  # (S,)
        q50 = (cdf >= 0.5).float().argmax(dim=-1)
        q90 = (cdf >= 0.9).float().argmax(dim=-1)
        expected_duration = (pmf * time_bins).sum(dim=-1)  # (S,)
        threshold = min(quick_sale_threshold_hours, n_time_bins)
        sale_probability = pmf[:, :threshold].sum(dim=-1)  # (S,)

        idx_now = df_item.index[mask_now]
        df_out.loc[idx_now, "prediction_q10"]   = q10[mask_now].float().numpy()
        df_out.loc[idx_now, "prediction_q50"]   = q50[mask_now].float().numpy()
        df_out.loc[idx_now, "prediction_q90"]   = q90[mask_now].float().numpy()
        df_out.loc[idx_now, "expected_duration"] = expected_duration[mask_now].numpy()
        df_out.loc[idx_now, "sale_probability"]  = sale_probability[mask_now].numpy()

    for col in ["buyout", "bid", "time_left", "listing_age", "prediction_q10", "prediction_q50", "prediction_q90", "expected_duration", "sale_probability"]:
        if col in df_out.columns:
            df_out[col] = np.round(df_out[col].astype(float), 2)

    return df_out
