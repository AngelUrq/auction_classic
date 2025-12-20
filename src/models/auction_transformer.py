import torch
import lightning as L
import torch.nn as nn
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import pandas as pd
from pathlib import Path

class AuctionTransformer(L.LightningModule):
    def __init__(
        self,
        input_size: int,
        n_items: int,
        n_contexts: int,
        n_bonuses: int,
        n_modtypes: int,
        embedding_dim: int = 128,
        d_model: int = 512,
        dim_feedforward: int = 2048,
        nhead: int = 4,
        num_layers: int = 4,
        dropout_p: float = 0.1,
        learning_rate: float = 1e-4,
        quantiles: list[float] = [0.1, 0.5, 0.9],
        classification_threshold_hours: float = 8.0,
        classification_loss_weight: float = 1.0,
        logging_interval: int = 1000,
        log_raw_batch_data: bool = False,
        log_step_predictions: bool = False,
        max_hours_back: int = 0,
        use_lr_scheduler: bool = True,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.input_projection = nn.Linear(input_size + embedding_dim, d_model)
        
        self.regression_projection = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, len(quantiles))
        )
        
        self.classification_projection = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(d_model * 2, 1)
        )

        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        self.context_embeddings = nn.Embedding(n_contexts, embedding_dim)
        self.bonus_embeddings = nn.Embedding(n_bonuses, embedding_dim)
        self.modifier_embeddings = nn.Embedding(n_modtypes, embedding_dim)
        self.hour_of_week_embeddings = nn.Embedding(24 * 7, embedding_dim)
        self.time_offset_embeddings = nn.Embedding(max_hours_back + 1, embedding_dim)

        # Item <- Context
        self.context_conditioning = nn.Sequential(
            nn.Linear(2 * embedding_dim, 2 * embedding_dim),
            nn.SiLU(),
            nn.Linear(2 * embedding_dim, 2 * embedding_dim)
        )

        # Bonuses <- (Item, Context)
        self.bonus_conditioning = nn.Sequential(
            nn.Linear(2 * embedding_dim, 2 * embedding_dim),
            nn.SiLU(),
            nn.Linear(2 * embedding_dim, 2 * embedding_dim)
        )

        # Item <- Modifiers
        self.modifier_conditioning = nn.Sequential(
            nn.Linear(2 * embedding_dim, 2 * embedding_dim),
            nn.SiLU(),
            nn.Linear(2 * embedding_dim, 2 * embedding_dim)
        )

        # Project scalar modifier values (already normalized) into embedding space
        self.modifier_value_projection = nn.Sequential(
            nn.Linear(1, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout_p
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.learning_rate = learning_rate
        self.logging_interval = logging_interval
        self.log_raw_batch_data = log_raw_batch_data
        self.log_step_predictions = log_step_predictions
        self.quantiles = quantiles
        self.use_lr_scheduler = use_lr_scheduler
        self.classification_threshold_hours = classification_threshold_hours
        self.classification_loss_weight = classification_loss_weight

    def forward(self, X):
        (auctions, item_index, contexts, bonus_lists, modifier_types, modifier_values, hour_of_week, time_offset) = X

        # Base embeddings
        item_embeddings = self.item_embeddings(item_index.long())                   # (B,S,D)
        context_embeddings = self.context_embeddings(contexts.long())               # (B,S,D)
        bonus_embeddings_full = self.bonus_embeddings(bonus_lists.long())           # (B,S,K,D)
        modifier_type_embeddings = self.modifier_embeddings(modifier_types.long())  # (B,S,M,D)
        hour_of_week_embeddings = self.hour_of_week_embeddings(hour_of_week.long()) # (B,S,D)
        time_offset_embeddings = self.time_offset_embeddings(time_offset.long())    # (B,S,D)

        # Masks
        attention_mask = (item_index == 0)                                          # (B,S)
        item_mask = (item_index != 0).float().unsqueeze(-1)                         # (B,S,1)
        context_mask = (contexts != 0).float().unsqueeze(-1)                        # (B,S,1)
        bonus_mask = (bonus_lists != 0).float().unsqueeze(-1)                       # (B,S,K,1)
        modifier_mask = (modifier_types != 0).float().unsqueeze(-1)                 # (B,S,M,1)

        # Apply masks
        item_embeddings = item_embeddings * item_mask
        context_embeddings = context_embeddings * context_mask
        auctions = auctions * item_mask

        # ------------------------------
        # Item <- Context conditioning
        # ------------------------------
        anchor_context = torch.cat([item_embeddings, context_embeddings], dim=-1)   # (B,S,2D)
        gamma_beta_context = self.context_conditioning(anchor_context)              # (B,S,2D)
        gamma_ctx, beta_ctx = torch.chunk(gamma_beta_context, 2, dim=-1)            # (B,S,D),(B,S,D)
        item_embeddings = gamma_ctx * item_embeddings + beta_ctx                    # (B,S,D)

        # -----------------------------------------------------
        # Bonuses conditioned by (conditioned item, context)
        # -----------------------------------------------------
        anchor_bonus = torch.cat([item_embeddings, context_embeddings], dim=-1)     # (B,S,2D)
        gamma_beta_bonus = self.bonus_conditioning(anchor_bonus)                    # (B,S,2D)
        gamma_bonus, beta_bonus = torch.chunk(gamma_beta_bonus, 2, dim=-1)          # (B,S,D),(B,S,D)
        gamma_bonus = gamma_bonus.unsqueeze(-2)                                     # (B,S,1,D)
        beta_bonus  = beta_bonus.unsqueeze(-2)                                      # (B,S,1,D)

        bonus_embeddings_conditioned = bonus_embeddings_full * gamma_bonus + beta_bonus
        bonus_sum = (bonus_embeddings_conditioned * bonus_mask).sum(dim=-2)         # (B,S,D)
        bonus_count = bonus_mask.sum(dim=-2).clamp_min(1e-6)                        # (B,S,1)
        bonus_embeddings_pooled = bonus_sum / bonus_count                           # (B,S,D)

        # ---------------------------------------------
        # Modifiers: type emb + projected scalar value
        # ---------------------------------------------
        modifier_values = modifier_values.unsqueeze(-1)                             # (B,S,M,1)
        modifier_value_vectors = self.modifier_value_projection(modifier_values)    # (B,S,M,D)
        modifier_vectors = modifier_type_embeddings + modifier_value_vectors        # (B,S,M,D)

        modifier_sum = (modifier_vectors * modifier_mask).sum(dim=-2)               # (B,S,D)
        modifier_count = modifier_mask.sum(dim=-2).clamp_min(1e-6)                  # (B,S,1)
        modifier_embeddings_pooled = modifier_sum / modifier_count                  # (B,S,D)

        # ---------------------------------------------
        # Item <- Modifier conditioning
        # ---------------------------------------------
        anchor_modifier = torch.cat([item_embeddings, modifier_embeddings_pooled], dim=-1)
        gamma_beta_modifier = self.modifier_conditioning(anchor_modifier)           # (B,S,2D)
        gamma_mod, beta_mod = torch.chunk(gamma_beta_modifier, 2, dim=-1)           # (B,S,D),(B,S,D)
        item_embeddings = gamma_mod * item_embeddings + beta_mod                    # (B,S,D)

        item_embeddings_conditioned = (
            item_embeddings
            + context_embeddings
            + bonus_embeddings_pooled
            + modifier_embeddings_pooled
            + hour_of_week_embeddings
            + time_offset_embeddings
        )                                                                           # (B,S,D)

        # Project auctions + single conditioned item vector
        features = torch.cat([auctions, item_embeddings_conditioned], dim=-1)       # (B,S,input_size + D)
        X = self.input_projection(features)                                         # (B,S,d_model)
        
        X = self.encoder(X, src_key_padding_mask=attention_mask.bool())
        
        # Compute both outputs
        regression_output = self.regression_projection(X)
        classification_output = self.classification_projection(X)

        return regression_output, classification_output

    def _compute_classification_loss_and_metrics(self, y_hat_classification, y, loss_mask):
        """Compute binary classification loss and metrics.
        
        Args:
            y_hat_classification: Classification logits (B, S, 1)
            y: Target durations in hours (B, S)
            loss_mask: Mask for valid items (B, S, 1)
            
        Returns:
            Dictionary with keys: loss, accuracy, precision, recall, f1
        """
        # Create binary labels: 1 if item lasts < threshold hours, 0 otherwise
        labels = (y < self.classification_threshold_hours).float().unsqueeze(-1)  # (B, S, 1)
        
        # Compute binary cross-entropy loss
        classification_loss = nn.functional.binary_cross_entropy_with_logits(
            y_hat_classification * loss_mask,
            labels * loss_mask,
            reduction='sum'
        ) / loss_mask.sum().clamp_min(1e-6)
        
        with torch.no_grad():
            # Get predictions (apply sigmoid and threshold at 0.5)
            predictions = (torch.sigmoid(y_hat_classification) > 0.5).float()  # (B, S, 1)
            
            # Apply mask to get valid predictions and labels
            valid_predictions = predictions * loss_mask
            valid_labels = labels * loss_mask
            
            # Compute metrics
            true_positives = (valid_predictions * valid_labels).sum()
            false_positives = (valid_predictions * (1 - valid_labels)).sum()
            false_negatives = ((1 - valid_predictions) * valid_labels).sum()
            true_negatives = ((1 - valid_predictions) * (1 - valid_labels)).sum()
            
            # Accuracy
            correct = (valid_predictions == valid_labels).float() * loss_mask
            accuracy = correct.sum() / loss_mask.sum().clamp_min(1e-6)
            
            # Precision: TP / (TP + FP)
            precision = true_positives / (true_positives + false_positives).clamp_min(1e-6)
            
            # Recall: TP / (TP + FN)
            recall = true_positives / (true_positives + false_negatives).clamp_min(1e-6)
            
            # F1: 2 * (precision * recall) / (precision + recall)
            f1 = 2 * (precision * recall) / (precision + recall).clamp_min(1e-6)
        
        return {
            'loss': classification_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


    def _compute_loss_and_metrics(self, y_hat_regression, y_hat_classification, y, item_index, current_hours, time_left, time_offset):
        """Compute loss and metrics for both training and validation.
        
        Args:
            y_hat_regression: Quantile predictions (B, S, Q)
            y_hat_classification: Classification logits (B, S, 1)
            
        Returns:
            Dictionary with keys: total_loss, pinball_loss, classification_loss, general_mae, 
            recent_listings_mae, new_listings_mae, accuracy, precision, recall, f1, mask
        """
        item_mask = (item_index != 0).float().unsqueeze(-1) # (B,S,1)
        
        # Only compute loss for current auctions (time_offset == 0)
        current_auctions_mask = (time_offset == 0).float().unsqueeze(-1) # (B,S,1)
        loss_mask = item_mask * current_auctions_mask # (B,S,1)

        current_hours = current_hours.unsqueeze(-1)  # (B,S,1) 
        time_left = time_left.unsqueeze(-1)      # (B,S,1)

        # ----- Pinball loss over all quantiles -----
        quantile_targets = torch.as_tensor(
            self.quantiles, device=y_hat_regression.device, dtype=y_hat_regression.dtype
        ).view(1, 1, -1)  # (1,1,Q)

        prediction_errors = y.unsqueeze(-1) - y_hat_regression  # (B, S, Q)
        quantile_losses = torch.maximum(
            quantile_targets * prediction_errors,
            (quantile_targets - 1.0) * prediction_errors
        )  # (B, S, Q)

        pinball_loss = (quantile_losses * loss_mask).sum() / (loss_mask).sum().clamp_min(1e-6)

        # ----- Classification loss and metrics -----
        cls_metrics = self._compute_classification_loss_and_metrics(
            y_hat_classification, y, loss_mask
        )
        
        # ----- Combine losses -----
        total_loss = pinball_loss + self.classification_loss_weight * cls_metrics['loss']

        # ----- Metrics computed on the median (tau=0.5) channel -----
        median_index = self.quantiles.index(0.5)
        median_predictions = y_hat_regression[..., median_index]  # (B, S)

        with torch.no_grad():
            # General MAE (in hours) - only for current auctions
            valid_items_sum = loss_mask.sum().clamp_min(1e-6)
            general_mae = torch.nn.functional.l1_loss(
                median_predictions.unsqueeze(-1) * loss_mask,
                y.unsqueeze(-1)                  * loss_mask,
                reduction='sum'
            ) / valid_items_sum

            # Recent listings: current_hours <= 12 and full duration == 48 - only for current auctions
            recent_mask = loss_mask * (current_hours <= 12.0).float() * (time_left == 48.0).float()
            recent_items_sum = recent_mask.sum().clamp_min(1e-6)
            recent_listings_mae = torch.nn.functional.l1_loss(
                median_predictions.unsqueeze(-1) * recent_mask,
                y.unsqueeze(-1)                  * recent_mask,
                reduction='sum'
            ) / recent_items_sum

            # Brand new listings: current_hours == 0 - only for current auctions
            new_mask = loss_mask * (current_hours == 0.0).float() * (time_left == 48.0).float()
            new_items_sum = new_mask.sum().clamp_min(1e-6)
            new_listings_mae = torch.nn.functional.l1_loss(
                median_predictions.unsqueeze(-1) * new_mask,
                y.unsqueeze(-1)                  * new_mask,
                reduction='sum'
            ) / new_items_sum

        return {
            'total_loss': total_loss,
            'pinball_loss': pinball_loss,
            'classification_loss': cls_metrics['loss'],
            'general_mae': general_mae,
            'recent_listings_mae': recent_listings_mae,
            'new_listings_mae': new_listings_mae,
            'accuracy': cls_metrics['accuracy'],
            'precision': cls_metrics['precision'],
            'recall': cls_metrics['recall'],
            'f1': cls_metrics['f1'],
            'mask': loss_mask
        }
        
    def _compute_gradient_norm(self):
        """Compute the total gradient norm across all parameters."""
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def _compute_interval_metrics(
        self, y_hat, y, item_mask, lower_quantile=0.1, upper_quantile=0.9
    ):
        """
        Compute coverage and mean interval width for [lower_quantile, upper_quantile].
        Assumes y_hat and y are in HOURS already (0..48).
        """
        lower_index = self.quantiles.index(lower_quantile)
        upper_index = self.quantiles.index(upper_quantile)

        lower_predictions = y_hat[..., lower_index]  # (B, S)
        upper_predictions = y_hat[..., upper_index]  # (B, S)

        inside_interval = ((y >= lower_predictions) & (y <= upper_predictions)).float().unsqueeze(-1)  # (B, S, 1)

        valid_items_sum = item_mask.sum().clamp_min(1e-6)
        coverage = (inside_interval * item_mask).sum() / valid_items_sum

        interval_width = (upper_predictions - lower_predictions).unsqueeze(-1)  # (B, S, 1), in HOURS
        mean_interval_width = (interval_width * item_mask).sum() / valid_items_sum  # hours

        return coverage.item(), mean_interval_width.item()

    def _compute_quantile_calibration(self, y_hat, y, item_mask):
        """
        For each tau in self.quantiles, compute observed fraction of targets <= predicted q_tau.
        """
        quantile_calibration = {}
        total_valid_items = item_mask.sum().clamp_min(1e-6)

        for tau in self.quantiles:
            tau_index = self.quantiles.index(tau)
            quantile_predictions = y_hat[..., tau_index]  # (B, S)
            targets_below_or_equal = ((y <= quantile_predictions).float().unsqueeze(-1) * item_mask).sum()
            quantile_calibration[float(tau)] = (targets_below_or_equal / total_valid_items).item()

        return quantile_calibration

    def _log_quantile_metrics(self, prefix, y_hat, y, item_mask, lower_quantile=0.1, upper_quantile=0.9):
        """
        Log coverage/width for [lower_quantile, upper_quantile] and per-quantile calibration.
        Assumes y_hat and y are in HOURS already.
        """ 
        coverage, mean_interval_width = self._compute_interval_metrics(
            y_hat, y, item_mask, lower_quantile=lower_quantile, upper_quantile=upper_quantile
        )
        
        on_step = True if prefix == 'train' else False
        on_epoch = True
        
        self.log(
            f"{prefix}/coverage_p{int(lower_quantile*100)}_p{int(upper_quantile*100)}",
            coverage, on_step=on_step, on_epoch=on_epoch, prog_bar=False, batch_size=1
        )
        self.log(
            f"{prefix}/width_p{int(lower_quantile*100)}_p{int(upper_quantile*100)}_hours",
            mean_interval_width, on_step=on_step, on_epoch=on_epoch, prog_bar=False, batch_size=1
        )

        quantile_calibration = self._compute_quantile_calibration(y_hat, y, item_mask)
        for tau, observed_fraction in quantile_calibration.items():
            self.log(
                f"{prefix}/quantile_calibration_{tau:.2f}",
                observed_fraction, on_step=on_step, on_epoch=on_epoch, prog_bar=False, batch_size=1
            )

    def _log_metrics(self, prefix, loss, general_mae, recent_listings_mae, new_listings_mae):
        """Log metrics for training or validation."""
        self.log(f'{prefix}/loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log(f'{prefix}/general_mae', general_mae, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log(f'{prefix}/recent_listings_mae', recent_listings_mae, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log(f'{prefix}/new_listings_mae', new_listings_mae, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)

    def training_step(self, batch, batch_idx):
        y_hat_regression, y_hat_classification = self((
            batch['auctions'], 
            batch['item_index'], 
            batch['contexts'], 
            batch['bonus_lists'], 
            batch['modifier_types'], 
            batch['modifier_values'], 
            batch['hour_of_week'],
            batch['time_offset'],
        ))
        
        metrics = self._compute_loss_and_metrics(
            y_hat_regression, y_hat_classification, batch['target'], batch['item_index'], batch['current_hours_raw'], batch['time_left_raw'], batch['time_offset']
        )

        self._log_metrics('train', metrics['total_loss'], metrics['general_mae'], metrics['recent_listings_mae'], metrics['new_listings_mae'])
        self.log('train/pinball_loss', metrics['pinball_loss'], on_step=True, on_epoch=True, prog_bar=False, batch_size=1)
        self.log('train/classification_loss', metrics['classification_loss'], on_step=True, on_epoch=True, prog_bar=False, batch_size=1)
        self.log('train/classification_accuracy', metrics['accuracy'], on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('train/classification_precision', metrics['precision'], on_step=True, on_epoch=True, prog_bar=False, batch_size=1)
        self.log('train/classification_recall', metrics['recall'], on_step=True, on_epoch=True, prog_bar=False, batch_size=1)
        self.log('train/classification_f1', metrics['f1'], on_step=True, on_epoch=True, prog_bar=False, batch_size=1)
        self.log('train/lr', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=True)
        
        if self.global_step % self.logging_interval == 0:
            grad_norm = self._compute_gradient_norm()
            self.log('train/grad_norm', grad_norm, on_step=True, on_epoch=True, prog_bar=True)
            self._log_quantile_metrics(prefix='train', y_hat=y_hat_regression, y=batch['target'], item_mask=metrics['mask'])
            
            if self.log_step_predictions:
                self._log_step_predictions(batch_idx, batch['target'], y_hat_regression, y_hat_classification, metrics['mask'], 'train')

        if self.global_step == 0 and self.log_raw_batch_data:
            self._log_raw_batch_data(batch_idx, batch, y_hat_regression, y_hat_classification, metrics['mask'], 'train')
        
        return metrics['total_loss']

    def validation_step(self, batch, batch_idx):
        y_hat_regression, y_hat_classification = self((
            batch['auctions'], 
            batch['item_index'], 
            batch['contexts'], 
            batch['bonus_lists'], 
            batch['modifier_types'], 
            batch['modifier_values'],
            batch['hour_of_week'],
            batch['time_offset'],
        ))
        
        metrics = self._compute_loss_and_metrics(
            y_hat_regression, y_hat_classification, batch['target'], batch['item_index'], batch['current_hours_raw'], batch['time_left_raw'], batch['time_offset']
        )

        self._log_metrics('val', metrics['total_loss'], metrics['general_mae'], metrics['recent_listings_mae'], metrics['new_listings_mae'])
        self.log('val/pinball_loss', metrics['pinball_loss'], on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log('val/classification_loss', metrics['classification_loss'], on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log('val/classification_accuracy', metrics['accuracy'], on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('val/classification_precision', metrics['precision'], on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log('val/classification_recall', metrics['recall'], on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log('val/classification_f1', metrics['f1'], on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self._log_quantile_metrics(prefix='val', y_hat=y_hat_regression, y=batch['target'], item_mask=metrics['mask'])

        if self.log_step_predictions:
            self._log_step_predictions(batch_idx, batch['target'], y_hat_regression, y_hat_classification, metrics['mask'], 'val')

        if self.global_step == 0 and self.log_raw_batch_data:
            self._log_raw_batch_data(batch_idx, batch, y_hat_regression, y_hat_classification, metrics['mask'], 'val')

        return metrics['total_loss']

    def _log_step_predictions(self, batch_idx, y, y_hat_regression, y_hat_classification, mask, step_type):
        """Log predictions for the current step to CSV files.
        
        Args:
            batch_idx: The index of the current batch
            y: Ground truth values
            y_hat_regression: Quantile predictions (B, S, Q) where Q is number of quantiles
            y_hat_classification: Classification logits (B, S, 1)
            mask: Validity mask for the predictions
            step_type: Either 'train' or 'val'
        """
        step_dir = Path(f"../generated/logs/{step_type}/step_{self.global_step}")
        step_dir.mkdir(parents=True, exist_ok=True)
        
        for batch_item in range(len(y)):
            targets = y[batch_item].detach().cpu()
            # Extract all quantile predictions
            q10_predictions = y_hat_regression[batch_item, :, 0].detach().cpu()  # 0.1 quantile
            q50_predictions = y_hat_regression[batch_item, :, 1].detach().cpu()  # 0.5 quantile (median)
            q90_predictions = y_hat_regression[batch_item, :, 2].detach().cpu()  # 0.9 quantile
            # Extract classification predictions (apply sigmoid to get probabilities)
            classification_probs = torch.sigmoid(y_hat_classification[batch_item, :, 0]).detach().cpu()
            validity_mask = mask[batch_item, :, 0].detach().cpu()
            
            data = {
                'Position': range(len(targets)),
                'Target': [float(t) for t in targets],
                'Prediction_Q10': [float(p) for p in q10_predictions],
                'Prediction_Q50': [float(p) for p in q50_predictions],
                'Prediction_Q90': [float(p) for p in q90_predictions],
                'Classification_Prob': [float(p) for p in classification_probs],
                'Status': ['Valid' if validity_mask[i] else 'Padding' for i in range(len(targets))]
            }
            df = pd.DataFrame(data)
            
            output_file = step_dir / f"batch_{batch_idx}_item_{batch_item}.csv"
            df.to_csv(output_file, index=False)

    def _log_raw_batch_data(self, batch_idx, batch, y_hat_regression, y_hat_classification, mask, step_type):
        """Log raw batch data to CSV files.
        
        Args:
            batch_idx: The index of the current batch
            batch: Input batch containing all features
            y_hat_regression: Quantile predictions
            y_hat_classification: Classification logits
            mask: Validity mask
            step_type: Either 'train' or 'val'
        """
        batch_dir = Path(f"../generated/logs/{step_type}/raw_batch_data")
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        auctions = batch['auctions']
        targets = batch['target'].detach().cpu().float()
        validity_mask = mask.detach().cpu().float()
        batch_size, seq_length, n_features = auctions.shape
        
        for batch_item in range(batch_size):
            sequence_features = auctions[batch_item].detach().cpu().float()
            sequence_target = targets[batch_item].detach().cpu().float()
            # Extract all quantile predictions
            q10_predictions = y_hat_regression[batch_item, :, 0].detach().cpu().float()  # 0.1 quantile
            q50_predictions = y_hat_regression[batch_item, :, 1].detach().cpu().float()  # 0.5 quantile (median)
            q90_predictions = y_hat_regression[batch_item, :, 2].detach().cpu().float()  # 0.9 quantile
            # Extract classification predictions (apply sigmoid to get probabilities)
            classification_probs = torch.sigmoid(y_hat_classification[batch_item, :, 0]).detach().cpu().float()
            
            df_sequence = pd.DataFrame(
                sequence_features,
                columns=[f'feature_{i}' for i in range(n_features)]
            )
            
            df_sequence['target'] = sequence_target
            df_sequence['prediction_q10'] = q10_predictions
            df_sequence['prediction_q50'] = q50_predictions
            df_sequence['prediction_q90'] = q90_predictions
            df_sequence['classification_prob'] = classification_probs
            df_sequence['is_valid'] = [validity_mask[batch_item, i, 0] for i in range(seq_length)]
            
            # Add additional features if available
            if 'current_hours_raw' in batch:
                df_sequence['current_hours'] = batch['current_hours_raw'][batch_item].detach().cpu().float()
            if 'time_left_raw' in batch:
                df_sequence['time_left'] = batch['time_left_raw'][batch_item].detach().cpu().float()
            
            sequence_file = batch_dir / f"raw_batch_{batch_idx}_sequence_{batch_item}.csv"
            df_sequence.to_csv(sequence_file, index=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        print(f'total steps: {self.trainer.estimated_stepping_batches}')
        
        if not self.use_lr_scheduler:
            return optimizer

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.05,
                div_factor=25,
                final_div_factor=10,
                anneal_strategy="cos"
            ),
            "interval": "step",
            "frequency": 1
        }
    
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
