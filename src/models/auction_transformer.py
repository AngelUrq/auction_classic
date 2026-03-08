import logging
import torch
import lightning as L
import torch.nn as nn

from src.losses import NLLSurvivalLoss

torch.set_float32_matmul_precision('high')

logger = logging.getLogger(__name__)


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
        n_buyout_ranks: int = 64,
        n_time_bins: int = 48,
        deephit_nll_weight: float = 0.8,
        deephit_ranking_sigma: float = 0.1,
        logging_interval: int = 1000,
        max_hours_back: int = 0,
        use_lr_scheduler: bool = True,
        lr_warmup_fraction: float = 0.05,
        lr_cosine_annealing: bool = True,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.input_projection = nn.Linear(input_size + embedding_dim, d_model)

        self.survival_head = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, n_time_bins),
        )

        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        self.context_embeddings = nn.Embedding(n_contexts, embedding_dim)
        self.bonus_embeddings = nn.Embedding(n_bonuses, embedding_dim)
        self.modifier_embeddings = nn.Embedding(n_modtypes, embedding_dim)
        self.buyout_rank_embeddings = nn.Embedding(n_buyout_ranks, embedding_dim)
        self.hour_of_week_embeddings = nn.Embedding(24 * 7, embedding_dim)
        self.snapshot_offset_embeddings = nn.Embedding(max_hours_back + 1, embedding_dim)

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

        self.nll_survival_loss = NLLSurvivalLoss()

        self.learning_rate = learning_rate
        self.logging_interval = logging_interval
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_warmup_fraction = lr_warmup_fraction
        self.lr_cosine_annealing = lr_cosine_annealing

    def forward(self, X):
        """Run the forward pass through embeddings, FiLM conditioning, and transformer encoder.

        Args:
            X: Tuple of (auction_features, item_index, contexts, bonus_ids,
               modifier_types, modifier_values, buyout_rank, hour_of_week, snapshot_offset)

        Returns:
            Survival logits of shape (B, S, n_time_bins)
        """
        (auction_features, item_index, contexts, bonus_ids, modifier_types, modifier_values, buyout_rank, hour_of_week, snapshot_offset) = X

        # Base embeddings
        item_embeddings = self.item_embeddings(item_index.long())                      # (B,S,D)
        context_embeddings = self.context_embeddings(contexts.long())                  # (B,S,D)
        bonus_embeddings_full = self.bonus_embeddings(bonus_ids.long())                # (B,S,K,D)
        modifier_type_embeddings = self.modifier_embeddings(modifier_types.long())     # (B,S,M,D)
        buyout_rank_embeddings = self.buyout_rank_embeddings(buyout_rank.clamp(0, self.hparams.n_buyout_ranks - 1).long())  # (B,S,D)
        hour_of_week_embeddings = self.hour_of_week_embeddings(hour_of_week.long())    # (B,S,D)
        snapshot_offset_embeddings = self.snapshot_offset_embeddings(snapshot_offset.long())  # (B,S,D)

        # Masks
        attention_mask = (item_index == 0)                                             # (B,S)
        item_mask = (item_index != 0).float().unsqueeze(-1)                            # (B,S,1)
        context_mask = (contexts != 0).float().unsqueeze(-1)                           # (B,S,1)
        bonus_mask = (bonus_ids != 0).float().unsqueeze(-1)                            # (B,S,K,1)
        modifier_mask = (modifier_types != 0).float().unsqueeze(-1)                    # (B,S,M,1)

        # Apply masks
        item_embeddings = item_embeddings * item_mask
        context_embeddings = context_embeddings * context_mask
        auction_features = auction_features * item_mask

        # ------------------------------
        # Item <- Context conditioning
        # ------------------------------
        anchor_context = torch.cat([item_embeddings, context_embeddings], dim=-1)      # (B,S,2D)
        gamma_beta_context = self.context_conditioning(anchor_context)                 # (B,S,2D)
        gamma_ctx, beta_ctx = torch.chunk(gamma_beta_context, 2, dim=-1)               # (B,S,D),(B,S,D)
        item_embeddings = gamma_ctx * item_embeddings + beta_ctx                       # (B,S,D)

        # -----------------------------------------------------
        # Bonuses conditioned by (conditioned item, context)
        # -----------------------------------------------------
        anchor_bonus = torch.cat([item_embeddings, context_embeddings], dim=-1)        # (B,S,2D)
        gamma_beta_bonus = self.bonus_conditioning(anchor_bonus)                       # (B,S,2D)
        gamma_bonus, beta_bonus = torch.chunk(gamma_beta_bonus, 2, dim=-1)             # (B,S,D),(B,S,D)
        gamma_bonus = gamma_bonus.unsqueeze(-2)                                        # (B,S,1,D)
        beta_bonus  = beta_bonus.unsqueeze(-2)                                         # (B,S,1,D)

        bonus_embeddings_conditioned = bonus_embeddings_full * gamma_bonus + beta_bonus
        bonus_sum = (bonus_embeddings_conditioned * bonus_mask).sum(dim=-2)            # (B,S,D)
        bonus_count = bonus_mask.sum(dim=-2).clamp_min(1e-6)                           # (B,S,1)
        bonus_embeddings_pooled = bonus_sum / bonus_count                              # (B,S,D)

        # ---------------------------------------------
        # Modifiers: type emb + projected scalar value
        # ---------------------------------------------
        modifier_values = modifier_values.unsqueeze(-1)                                # (B,S,M,1)
        modifier_value_vectors = self.modifier_value_projection(modifier_values)       # (B,S,M,D)
        modifier_vectors = modifier_type_embeddings + modifier_value_vectors           # (B,S,M,D)

        modifier_sum = (modifier_vectors * modifier_mask).sum(dim=-2)                  # (B,S,D)
        modifier_count = modifier_mask.sum(dim=-2).clamp_min(1e-6)                     # (B,S,1)
        modifier_embeddings_pooled = modifier_sum / modifier_count                     # (B,S,D)

        # ---------------------------------------------
        # Item <- Modifier conditioning
        # ---------------------------------------------
        anchor_modifier = torch.cat([item_embeddings, modifier_embeddings_pooled], dim=-1)
        gamma_beta_modifier = self.modifier_conditioning(anchor_modifier)              # (B,S,2D)
        gamma_mod, beta_mod = torch.chunk(gamma_beta_modifier, 2, dim=-1)              # (B,S,D),(B,S,D)
        item_embeddings = gamma_mod * item_embeddings + beta_mod                       # (B,S,D)

        item_embeddings_conditioned = (
            item_embeddings
            + context_embeddings
            + bonus_embeddings_pooled
            + modifier_embeddings_pooled
            + buyout_rank_embeddings
            + hour_of_week_embeddings
            + snapshot_offset_embeddings
        )                                                                              # (B,S,D)

        # Project auction_features + single conditioned item vector
        features = torch.cat([auction_features, item_embeddings_conditioned], dim=-1) # (B,S,input_size + D)
        X = self.input_projection(features)                                            # (B,S,d_model)

        X = self.encoder(X, src_key_padding_mask=attention_mask.bool())

        survival_logits = self.survival_head(X)                                        # (B,S,n_time_bins)
        return survival_logits

    def _compute_survival_loss(self, survival_logits, listing_duration, is_expired, item_index, snapshot_offset):
        """Compute survival loss over valid positions.

        Args:
            survival_logits: Raw logits from survival head (B, S, n_time_bins)
            listing_duration: Duration in hours, 0-47 (B, S)
            is_expired: 1 if auction expired (censored), 0 if sold (event) (B, S)
            item_index: Item indices for masking padding (B, S)
            snapshot_offset: Snapshot offset for selecting current auctions (B, S)

        Returns:
            Loss scalar
        """
        valid_mask = (item_index != 0) & (snapshot_offset == 0)

        if not valid_mask.any():
            return 0.0 * survival_logits.sum()

        logits = survival_logits[valid_mask]          # (N, n_time_bins)
        durations = listing_duration[valid_mask]       # (N,)
        events = 1.0 - is_expired[valid_mask].float()  # sold=1, expired=0

        return self.nll_survival_loss(logits, durations, events)

    def _compute_gradient_norm(self):
        """Compute the total gradient norm across all parameters."""
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def _log_metrics(self, prefix, loss):
        """Log metrics for training or validation."""
        on_step = (prefix == 'train')
        self.log(f'{prefix}/loss', loss, on_step=on_step, on_epoch=True, prog_bar=True, batch_size=1)

    def training_step(self, batch, batch_idx):
        """Compute survival loss and log training metrics."""
        survival_logits = self((
            batch['auction_features'],
            batch['item_index'],
            batch['contexts'],
            batch['bonus_ids'],
            batch['modifier_types'],
            batch['modifier_values'],
            batch['buyout_rank'],
            batch['hour_of_week'],
            batch['snapshot_offset'],
        ))

        loss = self._compute_survival_loss(
            survival_logits, batch['listing_duration'], batch['is_expired'],
            batch['item_index'], batch['snapshot_offset']
        )

        self._log_metrics('train', loss)
        self.log('train/lr', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=True)

        if self.global_step % self.logging_interval == 0:
            grad_norm = self._compute_gradient_norm()
            self.log('train/grad_norm', grad_norm, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Compute survival loss for validation."""
        survival_logits = self((
            batch['auction_features'],
            batch['item_index'],
            batch['contexts'],
            batch['bonus_ids'],
            batch['modifier_types'],
            batch['modifier_values'],
            batch['buyout_rank'],
            batch['hour_of_week'],
            batch['snapshot_offset'],
        ))

        loss = self._compute_survival_loss(
            survival_logits, batch['listing_duration'], batch['is_expired'],
            batch['item_index'], batch['snapshot_offset']
        )

        self._log_metrics('val', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        logger.info(f'Total steps: {self.trainer.estimated_stepping_batches}')

        if not self.use_lr_scheduler:
            return optimizer

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.lr_warmup_fraction,
                div_factor=25 if self.lr_warmup_fraction > 0 else 1,
                final_div_factor=10,
                anneal_strategy="cos" if self.lr_cosine_annealing else "linear",
            ),
            "interval": "step",
            "frequency": 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
