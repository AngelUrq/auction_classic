import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F


MAX_BONUSES = 9
MAX_MODIFIERS = 11

def predict_dataframe(model, df_auctions, prediction_time, feature_stats, lambda_value = 0.0401, max_hours_back = 0):
    model.eval()

    # only use rows the model can embed (context 0..max_hours_back in the past)
    df = df_auctions[(df_auctions["time_offset"] >= 0) & (df_auctions["time_offset"] <= max_hours_back)].copy()

    # sort by id and time_offset
    #df.sort_values(["id", "time_offset"], inplace=True)
    #df.to_csv("df_auctions.csv", index=False)

    # sequences per auction id (history of a single listing)
    grouped = {idx: g for idx, g in df.groupby("item_index")}

    df_out = df.copy()
    df_out["prediction_q10"] = np.nan
    df_out["prediction_q50"] = np.nan
    df_out["prediction_q90"] = np.nan
    df_out["sale_probability"] = np.nan

    skipped = set()

    for auction_id, df_item in tqdm(grouped.items(), desc="Inference per auction"):
        df_item = df_item.sort_values("time_offset")

        # -------- scalar features --------
        features_np = np.stack([
            np.log1p(df_item["bid"].to_numpy(dtype=np.float32)),
            np.log1p(df_item["buyout"].to_numpy(dtype=np.float32)),
            df_item["quantity"].to_numpy(dtype=np.float32),
            df_item["time_left"].to_numpy(dtype=np.float32),
            df_item["current_hours"].to_numpy(dtype=np.float32),
        ], axis=1)

        features = torch.tensor(features_np, dtype=torch.float32, device=model.device)
        features = (features - feature_stats["means"][:5].to(model.device)) / (feature_stats["stds"][:5].to(model.device) + 1e-6)

        # -------- categorical / set features --------
        item_indices = torch.tensor(df_item["item_index"].to_numpy(), dtype=torch.long, device=model.device)
        contexts = torch.tensor(df_item["context"].to_numpy(), dtype=torch.long, device=model.device)

        # bonuses: lists guaranteed; slice/pad to MAX_BONUSES
        bonus_lists_np = np.asarray([
            (row_bonuses[:MAX_BONUSES] + [0] * (MAX_BONUSES - len(row_bonuses)))
            for row_bonuses in df_item["bonus_lists"]
        ], dtype=np.int64)
        bonus_lists = torch.tensor(bonus_lists_np, dtype=torch.long, device=model.device)

        # modifiers: lists guaranteed; slice/pad to MAX_MODIFIERS
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

        # temporal inputs
        hour_of_week = torch.tensor(df_item["hour_of_week"].to_numpy(), dtype=torch.long, device=model.device)
        time_offset = torch.tensor(
            np.clip(df_item["time_offset"].to_numpy(), 0, max_hours_back),
            dtype=torch.int32,
            device=model.device
        )

        # pack (batch = 1)
        X = (
            features.unsqueeze(0),
            item_indices.unsqueeze(0),
            contexts.unsqueeze(0),
            bonus_lists.unsqueeze(0),
            modifier_types.unsqueeze(0),
            modifier_values.unsqueeze(0),
            hour_of_week.unsqueeze(0),
            time_offset.unsqueeze(0),
        )

        # (B, S, Q) -> (S, Q)
        y_pred_quantiles = model(X)[0]

        # write predictions only for "now" rows (time_offset == 0)
        mask_now = (df_item["time_offset"].to_numpy() == 0)
        if mask_now.any():
            q = y_pred_quantiles.detach().cpu().numpy()
            idx_now = df_item.index[mask_now]
            df_out.loc[idx_now, "prediction_q10"] = q[mask_now, 0]
            df_out.loc[idx_now, "prediction_q50"] = q[mask_now, 1]
            df_out.loc[idx_now, "prediction_q90"] = q[mask_now, 2]

            current_hours_now = df_item.loc[mask_now, "current_hours"].to_numpy(dtype=np.float32)
            sale_prob = np.exp(-lambda_value * (q[mask_now, 1] + current_hours_now))
            df_out.loc[idx_now, "sale_probability"] = sale_prob

    if skipped:
        df_out = df_out[~df_out["id"].isin(skipped)]

    # rounding for display (NaNs remain)
    for col in ["buyout","bid","time_left","current_hours","prediction_q10","prediction_q50","prediction_q90","sale_probability"]:
        if col in df_out.columns:
            df_out[col] = np.round(df_out[col].astype(float), 2)

    return df_out