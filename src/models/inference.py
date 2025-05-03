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
    df_out["prediction"]       = 0.0
    df_out["sale_probability"] = 0.0

    skipped = set()

    for item_idx, df_item in grouped.items():
        if len(df_item) > 64:
            skipped.add(item_idx)
            continue

        raw_buyout = df_item["buyout"].to_numpy(dtype=np.float32)

        feats = []
        for _, row in df_item.iterrows():
            bid        = np.log1p(row["bid"])
            buyout     = np.log1p(row["buyout"])
            quantity   = row["quantity"]
            time_left  = row["time_left"]
            cur_hours  = row["current_hours"]

            hour_sin     = np.sin(2 * np.pi * prediction_time.hour / 24)
            weekday_sin  = np.sin(2 * np.pi * prediction_time.weekday() / 7)

            feats.append([bid, buyout, quantity, time_left, cur_hours, hour_sin, weekday_sin])

        feats = torch.tensor(feats, dtype=torch.float32, device=model.device)
        feats = (feats - feature_stats["means"].to(model.device)) / \
                (feature_stats["stds"].to(model.device) + 1e-6)

        item_indices = torch.tensor(df_item["item_index"].to_numpy(), dtype=torch.int32, device=model.device)

        contexts = torch.tensor(df_item["context"].to_numpy(), dtype=torch.int32, device=model.device)

        bonus_lists     = [torch.tensor(x, dtype=torch.int32) for x in df_item["bonus_lists"]]
        modifier_types  = [torch.tensor(x, dtype=torch.int32) for x in df_item["modifier_types"]]
        modifier_values = [torch.tensor(np.log1p(x), dtype=torch.float32) for x in df_item["modifier_values"]]

        bonus_lists     = pad_tensors_to_max_size(bonus_lists).to(model.device)
        modifier_types  = pad_tensors_to_max_size(modifier_types).to(model.device)
        modifier_values = pad_tensors_to_max_size(modifier_values).to(model.device)
        modifier_values = (modifier_values - feature_stats["modifiers_mean"].to(model.device)) / (feature_stats["modifiers_std"].to(model.device) + 1e-6)

        buyout_for_rank          = raw_buyout.copy()
        buyout_for_rank[buyout_for_rank == 0] = np.inf
        uniq_prices              = np.unique(buyout_for_rank[buyout_for_rank != np.inf])
        ranks                    = np.zeros_like(raw_buyout, dtype=int)
        for r, p in enumerate(uniq_prices, 1):
            ranks[buyout_for_rank == p] = r

        buyout_ranking = torch.tensor(ranks, dtype=torch.int32, device=model.device)

        X = (feats.unsqueeze(0), item_indices.unsqueeze(0), contexts.unsqueeze(0),
             bonus_lists.unsqueeze(0), modifier_types.unsqueeze(0),
             modifier_values.unsqueeze(0), buyout_ranking.unsqueeze(0))

        y_pred = model(X)[0, :, 0] * 48.0

        df_out.loc[df_item.index, "prediction"] = y_pred.cpu().numpy()
        df_out.loc[df_item.index, "sale_probability"] = np.exp(-lambda_value * y_pred.cpu().numpy())

    if skipped:
        df_out = df_out[~df_out["item_index"].isin(skipped)]

    return df_out
