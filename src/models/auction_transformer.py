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
        logging_interval: int = 1000
    ):
        super().__init__()

        self.save_hyperparameters()

        self.input_projection = nn.Linear(input_size + 5 * embedding_dim, d_model)
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
        
        self.criterion = torch.nn.MSELoss(reduction='none')
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

        combined_features = torch.cat([
            auctions, 
            item_embeddings, 
            context_embeddings, 
            bonus_embeddings, 
            modifier_embeddings,
        ], dim=-1)

        X = self.input_projection(combined_features)
        
        attention_mask = (item_index == 0)
        X = self.encoder(X, src_key_padding_mask=attention_mask)
        
        X = self.output_projection(X)
        X = torch.sigmoid(X)
        
        return X

    def _compute_loss_and_metrics(self, y_hat, y, item_index, current_hours, time_left):
        """Compute loss and metrics for both training and validation."""
        mask = (item_index != 0).float().unsqueeze(-1)
        weights = torch.exp(-current_hours / 24.0).unsqueeze(-1)
        
        # Compute MSE loss
        errors = self.criterion(y_hat, y.unsqueeze(2))  
        weighted_errors = errors * mask * weights
        mse_loss = weighted_errors.sum() / (mask * weights).sum()
        
        with torch.no_grad():
            # Calculate MAE for all valid items
            general_mae = torch.nn.functional.l1_loss(
                y_hat * mask * 48.0, 
                y.unsqueeze(2) * mask * 48.0, 
                reduction='sum'
            ) / mask.sum()
            
            # Calculate MAE for recent listings with full duration
            recent_listings_mask = mask * (current_hours <= 12.0).float().unsqueeze(-1) * (time_left == 48.0).float().unsqueeze(-1)
            recent_listings_mae = torch.nn.functional.l1_loss(
                y_hat * recent_listings_mask * 48.0, 
                y.unsqueeze(2) * recent_listings_mask * 48.0, 
                reduction='sum'
            ) / (recent_listings_mask.sum() + 1e-6)
            
            # Calculate MAE for brand new listings (current_hours = 0)
            new_listings_mask = mask * (current_hours == 0.0).float().unsqueeze(-1)
            new_listings_mae = torch.nn.functional.l1_loss(
                y_hat * new_listings_mask * 48.0, 
                y.unsqueeze(2) * new_listings_mask * 48.0, 
                reduction='sum'
            ) / (new_listings_mask.sum() + 1e-6)
            
        return mse_loss, general_mae, recent_listings_mae, new_listings_mae, mask
    
    def _log_metrics(self, prefix, mse_loss, general_mae, recent_listings_mae, new_listings_mae):
        """Log metrics for training or validation."""
        self.log(f'{prefix}/mse_loss', mse_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log(f'{prefix}/general_mae', general_mae, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log(f'{prefix}/recent_listings_mae', recent_listings_mae, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log(f'{prefix}/new_listings_mae', new_listings_mae, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)

    def training_step(self, batch, batch_idx):
        y_hat = self((
            batch['auctions'], 
            batch['item_index'], 
            batch['contexts'], 
            batch['bonus_lists'], 
            batch['modifier_types'], 
            batch['modifier_values'], 
        ))
        
        mse_loss, general_mae, recent_listings_mae, new_listings_mae, mask = self._compute_loss_and_metrics(
            y_hat, batch['target'], batch['item_index'], batch['current_hours_raw'], batch['time_left_raw']
        )

        self._log_metrics('train', mse_loss, general_mae, recent_listings_mae, new_listings_mae)
        self.log('train/lr', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=True)
        
        return mse_loss

    def validation_step(self, batch, batch_idx):
        y_hat = self((
            batch['auctions'], 
            batch['item_index'], 
            batch['contexts'], 
            batch['bonus_lists'], 
            batch['modifier_types'], 
            batch['modifier_values'],
        ))
        
        mse_loss, general_mae, recent_listings_mae, new_listings_mae, mask = self._compute_loss_and_metrics(
            y_hat, batch['target'], batch['item_index'], batch['current_hours_raw'], batch['time_left_raw']
        )

        self._log_metrics('val', mse_loss, general_mae, recent_listings_mae, new_listings_mae)

        return mse_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
                div_factor=25,
                final_div_factor=100,
                anneal_strategy="cos"
            ),
            "interval": "step",
            "frequency": 1
        }
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}