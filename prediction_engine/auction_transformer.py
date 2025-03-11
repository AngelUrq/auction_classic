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
        embedding_dim: int = 128,
        d_model: int = 512,
        dim_feedforward: int = 2048,
        nhead: int = 4,
        num_layers: int = 4,
        dropout_p: float = 0.1,
        learning_rate: float = 3e-5,
        max_seq_len: int = 64,
        logging_interval: int = 1000
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_projection = nn.Linear(input_size - 1 + embedding_dim, d_model)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, 1)
        )
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout_p
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Compute sinusoidal positional encoding
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', pe)
        
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.learning_rate = learning_rate
        self.logging_interval = logging_interval
        
    def forward(self, X):
        seq_len = X.size(1)

        item_ids = X[:, :, 0].long()
        item_features = X[:, :, 1:]
        item_embeddings = self.item_embeddings(item_ids)
        
        combined_features = torch.cat([item_features, item_embeddings], dim=-1)
        
        X = self.input_projection(combined_features)
        X = X + self.positional_encoding[:seq_len, :].unsqueeze(0)
        
        X = self.encoder(X)
        X = self.output_projection(X)
        X = torch.sigmoid(X)
        
        return X

    def _log_step_predictions(self, batch_idx, y, y_hat, lengths, step_type):
        """Log predictions for the current step to CSV files.
        
        Args:
            batch_idx: The index of the current batch
            y: Ground truth values
            y_hat: Model predictions
            lengths: Sequence lengths
            step_type: Either 'train' or 'val'
        """
        step_dir = Path(f"logs/{step_type}/step_{self.global_step}")
        step_dir.mkdir(parents=True, exist_ok=True)
        
        for batch_item in range(len(y)):
            targets = y[batch_item].detach().cpu()
            predictions = y_hat[batch_item, :, 0].detach().cpu()
            length = lengths[batch_item].cpu().item()
            
            data = {
                'Position': range(len(targets)),
                'Target': [float(t * 48.0) for t in targets],
                'Prediction': [float(p * 48.0) for p in predictions],
                'Status': ['Valid' if i < length else 'Padding' for i in range(len(targets))]
            }
            df = pd.DataFrame(data)
            
            output_file = step_dir / f"batch_{batch_idx}_item_{batch_item}.csv"
            df.to_csv(output_file, index=False)

    def _log_raw_batch_data(self, batch_idx, x, y, lengths, step_type):
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
        
        features = x.detach().cpu()
        targets = y.detach().cpu() * 48.0
        lengths = lengths.cpu()
        batch_size, seq_length, n_features = features.shape
        
        for batch_item in range(batch_size):
            sequence_features = features[batch_item]
            sequence_target = targets[batch_item]
            sequence_length = lengths[batch_item]
            
            df_sequence = pd.DataFrame(
                sequence_features,
                columns=[f'feature_{i}' for i in range(n_features)]
            )
            
            df_sequence['target'] = sequence_target
            df_sequence['is_valid'] = [i < sequence_length for i in range(seq_length)]
            
            sequence_file = batch_dir / f"raw_batch_{batch_idx}_sequence_{batch_item}.csv"
            df_sequence.to_csv(sequence_file, index=False)

    def training_step(self, batch, batch_idx):
        x, y, lengths = batch
        y_hat = self(x)
        
        batch_size, seq_length = y.shape
        mask = torch.arange(seq_length, device=y.device).expand(batch_size, seq_length) < torch.tensor(lengths, device=y.device).unsqueeze(1)
        mask = mask.float().unsqueeze(2)
        
        mse_loss = self.criterion(y_hat * mask, y.unsqueeze(2) * mask) / mask.sum()

        with torch.no_grad():
            mae_loss = torch.nn.functional.l1_loss(
                y_hat * mask * 48.0, 
                y.unsqueeze(2) * mask * 48.0, 
                reduction='sum'
            ) / mask.sum()
        
        self.log('train/mse_loss', mse_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('train/mae_loss', mae_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('train/lr', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=True)
        
        if self.global_step % self.logging_interval == 0:
            self._log_step_predictions(batch_idx, y, y_hat, lengths, 'train')

        if self.global_step == 0:
            self._log_raw_batch_data(batch_idx, x, y, lengths, 'train')
        
        return mse_loss

    def validation_step(self, batch, batch_idx):
        x, y, lengths = batch
        y_hat = self(x)
        
        batch_size, seq_length = y.shape
        mask = torch.arange(seq_length, device=y.device).expand(batch_size, seq_length) < torch.tensor(lengths, device=y.device).unsqueeze(1)
        mask = mask.float().unsqueeze(2)

        mse_loss = self.criterion(y_hat * mask, y.unsqueeze(2) * mask) / mask.sum()
        mae_loss = torch.nn.functional.l1_loss(
            y_hat * mask * 48.0, 
            y.unsqueeze(2) * mask * 48.0, 
            reduction='sum'
        ) / mask.sum()
        
        self.log('val/mse_loss', mse_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('val/mae_loss', mae_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)

        if self.global_step % self.logging_interval == 0:
            self._log_step_predictions(batch_idx, y, y_hat, lengths, 'val')

        if self.global_step == 0:
            self._log_raw_batch_data(batch_idx, x, y, lengths, 'val')

        return mse_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=2500,
            verbose=True,
            min_lr=1e-6
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
