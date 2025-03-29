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
        learning_rate: float = 3e-5,
        logging_interval: int = 1000
    ):
        super().__init__()

        self.save_hyperparameters()

        self.input_projection = nn.Linear(input_size + 4 * embedding_dim, d_model)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, 1)
        )

        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        self.context_embeddings = nn.Embedding(n_contexts, embedding_dim)
        self.bonus_embeddings = nn.Embedding(n_bonuses, embedding_dim)
        self.modifier_embeddings = nn.Embedding(n_modtypes, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout_p
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.learning_rate = learning_rate
        self.logging_interval = logging_interval
        
    def forward(self, X):
        (auctions, item_index, contexts, bonus_lists, modifier_types, modifier_values) = X

        item_embeddings = self.item_embeddings(item_index.long())
        context_embeddings = self.context_embeddings(contexts.long())
        bonus_embeddings = self.bonus_embeddings(bonus_lists.long()) 
        modifier_embeddings = self.modifier_embeddings(modifier_types.long())

        item_mask = (item_index != 0).float().unsqueeze(-1)
        item_embeddings = item_embeddings * item_mask

        auctions = auctions * item_mask

        context_mask = (contexts != 0).float().unsqueeze(-1)
        context_embeddings = context_embeddings * context_mask

        bonus_mask = (bonus_lists != 0).float().unsqueeze(-1)
        bonus_embeddings = torch.sum(bonus_embeddings * bonus_mask, dim=-2) / (bonus_mask.sum(dim=-2) + 1e-6)

        modifier_mask = (modifier_types != 0).float().unsqueeze(-1)
        modifier_embeddings = modifier_values.unsqueeze(-1) * modifier_embeddings
        modifier_embeddings = torch.sum(modifier_embeddings * modifier_mask, dim=-2) / (modifier_mask.sum(dim=-2) + 1e-6)

        combined_features = torch.cat([auctions, item_embeddings, context_embeddings, bonus_embeddings, modifier_embeddings], dim=-1)
        X = self.input_projection(combined_features)
        
        attention_mask = (item_index == 0)
        X = self.encoder(X, src_key_padding_mask=attention_mask)
        
        X = self.output_projection(X)
        X = torch.sigmoid(X)
        
        return X

    def training_step(self, batch, batch_idx):
        (auctions, item_index, contexts, bonus_lists, modifier_types, modifier_values, current_hours), y = batch

        y_hat = self((auctions, item_index, contexts, bonus_lists, modifier_types, modifier_values))
        
        mask = (item_index != 0).float().unsqueeze(-1)

        weights = torch.exp(-current_hours / 24.0).unsqueeze(-1)
        mse = self.criterion(y_hat * mask, y.unsqueeze(2) * mask)
        weighted_mse = mse * weights * mask
        mse_loss = weighted_mse.sum() / (mask * weights).sum()

        with torch.no_grad():
            current_hours_mask = (current_hours <= 12.0).float().unsqueeze(-1)
            mask = mask * current_hours_mask
            mae_loss = torch.nn.functional.l1_loss(
                y_hat * mask * 48.0, 
                y.unsqueeze(2) * mask * 48.0, 
                reduction='sum'
            ) / mask.sum()
        
        self.log('train/mse_loss', mse_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('train/mae_loss', mae_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('train/lr', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=True)
        
        if self.global_step % self.logging_interval == 0:
            self._log_step_predictions(batch_idx, y, y_hat, mask, 'train')

        if self.global_step == 0:
            self._log_raw_batch_data(batch_idx, (auctions, item_index, contexts, bonus_lists, modifier_types, modifier_values, current_hours), y, mask, 'train')
        
        return mse_loss

    def validation_step(self, batch, batch_idx):
        (auctions, item_index, contexts, bonus_lists, modifier_types, modifier_values, current_hours), y = batch

        y_hat = self((auctions, item_index, contexts, bonus_lists, modifier_types, modifier_values))
        
        mask = (item_index != 0).float().unsqueeze(-1)

        weights = torch.exp(-current_hours / 24.0).unsqueeze(-1)
        mse = self.criterion(y_hat * mask, y.unsqueeze(2) * mask)
        weighted_mse = mse * weights * mask
        mse_loss = weighted_mse.sum() / (mask * weights).sum()

        current_hours_mask = (current_hours <= 12.0).float().unsqueeze(-1)
        mask = mask * current_hours_mask
        mae_loss = torch.nn.functional.l1_loss(
            y_hat * mask * 48.0, 
            y.unsqueeze(2) * mask * 48.0, 
            reduction='sum'
        ) / mask.sum()
        
        self.log('val/mse_loss', mse_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('val/mae_loss', mae_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)

        self._log_step_predictions(batch_idx, y, y_hat, mask, 'val')

        if self.global_step == 0:
            self._log_raw_batch_data(batch_idx, (auctions, item_index, contexts, bonus_lists, modifier_types, modifier_values, current_hours), y, mask, 'val')

        return mse_loss

    def _log_step_predictions(self, batch_idx, y, y_hat, mask, step_type):
        """Log predictions for the current step to CSV files.
        
        Args:
            batch_idx: The index of the current batch
            y: Ground truth values
            y_hat: Model predictions
            step_type: Either 'train' or 'val'
        """
        step_dir = Path(f"logs/{step_type}/step_{self.global_step}")
        step_dir.mkdir(parents=True, exist_ok=True)
        
        for batch_item in range(len(y)):
            targets = y[batch_item].detach().cpu()
            predictions = y_hat[batch_item, :, 0].detach().cpu()
            
            data = {
                'Position': range(len(targets)),
                'Target': [float(t * 48.0) for t in targets],
                'Prediction': [float(p * 48.0) for p in predictions],
                'Status': ['Valid' if mask[batch_item, i] else 'Padding' for i in range(len(targets))]
            }
            df = pd.DataFrame(data)
            
            output_file = step_dir / f"batch_{batch_idx}_item_{batch_item}.csv"
            df.to_csv(output_file, index=False)

    def _log_raw_batch_data(self, batch_idx, x, y, mask, step_type):
        """Log raw batch data to CSV files.
        
        Args:
            batch_idx: The index of the current batch
            x: Input features
            y: Ground truth values
            lengths: Sequence lengths
            step_type: Either 'train' or 'val'
        """
        batch_dir = Path(f"logs/{step_type}/raw_batch_data")
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        (auctions, item_index, contexts, bonus_lists, modifier_types, modifier_values, current_hours) = x
        targets = y.detach().cpu() * 48.0
        mask = mask.detach().cpu()
        batch_size, seq_length, n_features = auctions.shape
        
        for batch_item in range(batch_size):
            sequence_features = auctions[batch_item].detach().cpu()
            sequence_target = targets[batch_item].detach().cpu()
            
            df_sequence = pd.DataFrame(
                sequence_features,
                columns=[f'feature_{i}' for i in range(n_features)]
            )
            
            df_sequence['target'] = sequence_target
            df_sequence['is_valid'] = [mask[batch_item, i] for i in range(seq_length)]
            
            sequence_file = batch_dir / f"raw_batch_{batch_idx}_sequence_{batch_item}.csv"
            df_sequence.to_csv(sequence_file, index=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=10000,
            verbose=True,
            min_lr=1e-8
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/mse_loss",
                "interval": "step",
                "frequency": 1
            },
        }   
