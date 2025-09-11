import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from src.data.utils import pad_tensors_to_max_size


@torch.no_grad()
def predict_dataframe(model, df_auctions, prediction_time, feature_stats, lambda_value = 0.0401):
    model.eval()

    grouped = {idx: df for idx, df in df_auctions.groupby("item_index")}

    df_out = df_auctions.copy()
    df_out["prediction_q10"] = 0.0
    df_out["prediction_q50"] = 0.0
    df_out["prediction_q90"] = 0.0
    df_out["sale_probability"] = 0.0

    skipped = set()

    for item_idx, df_item in grouped.items():
        if len(df_item) > 64:
            skipped.add(item_idx)
            continue

        raw_buyout = df_item["buyout"].to_numpy(dtype=np.float32)

        feats = []
        for _, row in df_item.iterrows():
            bid = np.log1p(row["bid"])
            buyout = np.log1p(row["buyout"])
            quantity = row["quantity"]
            time_left = row["time_left"]
            cur_hours = row["current_hours"]

            hour_sin = np.sin(2 * np.pi * prediction_time.hour / 24)
            hour_cos = np.cos(2 * np.pi * prediction_time.hour / 24)
            weekday_sin = np.sin(2 * np.pi * prediction_time.weekday() / 7)
            weekday_cos = np.cos(2 * np.pi * prediction_time.weekday() / 7)

            feats.append([bid, buyout, quantity, time_left, cur_hours, hour_sin, hour_cos, weekday_sin, weekday_cos])

        feats = torch.tensor(feats, dtype=torch.float32, device=model.device)
        feats = (feats - feature_stats["means"].to(model.device)) / \
                (feature_stats["stds"].to(model.device) + 1e-6)

        item_indices = torch.tensor(df_item["item_index"].to_numpy(), dtype=torch.int32, device=model.device)

        contexts = torch.tensor(df_item["context"].to_numpy(), dtype=torch.int32, device=model.device)

        bonus_lists = [torch.tensor(x, dtype=torch.int32) for x in df_item["bonus_lists"]]
        modifier_types = [torch.tensor(x, dtype=torch.int32) for x in df_item["modifier_types"]]
        modifier_values = [torch.tensor(np.log1p(x), dtype=torch.float32) for x in df_item["modifier_values"]]

        bonus_lists = pad_tensors_to_max_size(bonus_lists).to(model.device)
        modifier_types = pad_tensors_to_max_size(modifier_types).to(model.device)
        modifier_values = pad_tensors_to_max_size(modifier_values).to(model.device)
        modifier_values = (modifier_values - feature_stats["modifiers_mean"].to(model.device)) / (feature_stats["modifiers_std"].to(model.device) + 1e-6)

        X = (feats.unsqueeze(0), item_indices.unsqueeze(0), contexts.unsqueeze(0), bonus_lists.unsqueeze(0), modifier_types.unsqueeze(0), modifier_values.unsqueeze(0))

        # Model now returns quantile predictions: (batch_size, seq_length, num_quantiles)
        y_pred_quantiles = model(X)[0]  # Shape: (seq_length, num_quantiles)

        current_hours = df_item["current_hours"].to_numpy()
        
        # Extract quantile predictions [0.1, 0.5, 0.9]
        q10_pred = y_pred_quantiles[:, 0].cpu().numpy()  # 0.1 quantile
        q50_pred = y_pred_quantiles[:, 1].cpu().numpy()  # 0.5 quantile (median)
        q90_pred = y_pred_quantiles[:, 2].cpu().numpy()  # 0.9 quantile
        
        df_out.loc[df_item.index, "prediction_q10"] = q10_pred
        df_out.loc[df_item.index, "prediction_q50"] = q50_pred
        df_out.loc[df_item.index, "prediction_q90"] = q90_pred
        
        # Use median (q50) for sale probability calculation
        sale_prob = np.exp(-lambda_value * (q50_pred + current_hours))
        df_out.loc[df_item.index, "sale_probability"] = sale_prob

    if skipped:
        df_out = df_out[~df_out["item_index"].isin(skipped)]

    # Round columns that will be displayed in the UI to 2 decimal places
    df_out["buyout"] = np.round(df_out["buyout"], 2)
    df_out["bid"] = np.round(df_out["bid"], 2)
    df_out["time_left"] = np.round(df_out["time_left"], 2)
    df_out["current_hours"] = np.round(df_out["current_hours"], 2)
    df_out["prediction_q10"] = np.round(df_out["prediction_q10"], 2)
    df_out["prediction_q50"] = np.round(df_out["prediction_q50"], 2)
    df_out["prediction_q90"] = np.round(df_out["prediction_q90"], 2)
    df_out["sale_probability"] = np.round(df_out["sale_probability"], 2)

    return df_out
