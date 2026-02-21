import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F


MAX_BONUSES = 9
MAX_MODIFIERS = 11

def predict_dataframe(model, df_auctions, prediction_time, feature_stats, max_hours_back = 0, max_sequence_length = 4096):
    model.eval()

    df = df_auctions[(df_auctions["snapshot_offset"] >= 0) & (df_auctions["snapshot_offset"] <= max_hours_back)].copy()
    grouped = {idx: g for idx, g in df.groupby("item_index")}

    df_out = df.copy()
    df_out["prediction_q10"] = np.nan
    df_out["prediction_q50"] = np.nan
    df_out["prediction_q90"] = np.nan
    df_out["is_short_duration"] = np.nan

    for auction_id, df_item in grouped.items():
        # Keep the most recent entries (snapshot_offset closer to 0) and maintain
        # the same old->new ordering used during training.
        df_item = df_item.sort_values("snapshot_offset", ascending=False)
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
        contexts = torch.tensor(df_item["context"].to_numpy(), dtype=torch.long, device=model.device)

        bonus_ids_np = np.asarray([
            (row_bonuses[:MAX_BONUSES] + [0] * (MAX_BONUSES - len(row_bonuses)))
            for row_bonuses in df_item["bonus_ids"]
        ], dtype=np.int64)
        bonus_ids = torch.tensor(bonus_ids_np, dtype=torch.long, device=model.device)

        modifier_types_np = np.asarray([
            (row_types[:MAX_MODIFIERS] + [0] * (MAX_MODIFIERS - len(row_types)))
            for row_types in df_item["modifier_types"]
        ], dtype=np.int64)
        modifier_values_np = np.asarray([
            (row_vals[:MAX_MODIFIERS] + [0.0] * (MAX_MODIFIERS - len(row_vals)))
            for row_vals in df_item["modifier_values"]
        ], dtype=np.float32)

        modifier_types = torch.tensor(modifier_types_np, dtype=torch.long, device=model.device)
        modifier_values = torch.tensor(modifier_values_np, dtype=torch.float32, device=model.device)
        modifier_values = torch.log1p(modifier_values)
        modifier_values = (modifier_values - feature_stats["modifiers_mean"].to(model.device)) / (feature_stats["modifiers_std"].to(model.device) + 1e-6)

        hour_of_week = torch.tensor(df_item["hour_of_week"].to_numpy(), dtype=torch.long, device=model.device)
        snapshot_offset = torch.tensor(
            np.clip(df_item["snapshot_offset"].to_numpy(), 0, max_hours_back),
            dtype=torch.int32,
            device=model.device
        )

        X = (
            features.unsqueeze(0),
            item_indices.unsqueeze(0),
            contexts.unsqueeze(0),
            bonus_ids.unsqueeze(0),
            modifier_types.unsqueeze(0),
            modifier_values.unsqueeze(0),
            hour_of_week.unsqueeze(0),
            snapshot_offset.unsqueeze(0),
        )

        y_pred_quantiles, y_pred_classification = model(X)

        mask_now = (df_item["snapshot_offset"].to_numpy() == 0)
        if mask_now.any():
            q = y_pred_quantiles.detach().cpu().numpy()
            # Apply sigmoid to classification logits to get probability of short duration
            c = torch.sigmoid(y_pred_classification).detach().cpu().numpy()
            idx_now = df_item.index[mask_now]
            df_out.loc[idx_now, "prediction_q10"] = q[0, mask_now, 0]
            df_out.loc[idx_now, "prediction_q50"] = q[0, mask_now, 1]
            df_out.loc[idx_now, "prediction_q90"] = q[0, mask_now, 2]
            df_out.loc[idx_now, "is_short_duration"] = c[0, mask_now, 0]

            listing_age_now = df_item.loc[mask_now, "listing_age"].to_numpy(dtype=np.float32)

    for col in ["buyout","bid","time_left","listing_age","prediction_q10","prediction_q50","prediction_q90","is_short_duration"]:
        if col in df_out.columns:
            df_out[col] = np.round(df_out[col].astype(float), 2)

    return df_out
