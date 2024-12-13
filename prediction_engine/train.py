import os
import wandb
from tqdm import tqdm
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

def save_checkpoint(model, optimizer, epoch, iters, checkpoint_path='checkpoints'):
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_path, f"checkpoint_epoch_{epoch}_iter_{iters}.pt")
    torch.save({
        'epoch': epoch,
        'iter': iters,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_file)
    print(f"Checkpoint saved at {checkpoint_file}")

def train(
    model,
    train_loader,
    val_loader,
    epochs,
    eval_steps,
    device,
    optimizer,
    criterion,
    lr_scheduler
):
    print("Starting training for", epochs, "epochs")

    for epoch in tqdm(range(epochs)):
        model.train()

        mse_losses = []
        mae_losses = []
        
        for i, (X, y, lengths) in enumerate(tqdm(train_loader, total=len(train_loader))):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_pred = model(X)
            mask = (y != 0).float().unsqueeze(2)

            loss = criterion(y_pred * mask, y.unsqueeze(2)) / mask.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                mae = F.l1_loss(y_pred * mask, y.unsqueeze(2) * mask, reduction='sum') / mask.sum()

            mse_losses.append(loss.item())
            mae_losses.append(mae.item())

            if i % 50 == 0:
                mse_loss_avg = np.mean(mse_losses)
                mae_loss_avg = np.mean(mae_losses)
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch} Iteration {i} Loss {mse_loss_avg} MAE {mae_loss_avg} LR {lr}")
                lr_scheduler.step(mse_loss_avg)

                wandb.log({
                  "train/mse_loss": mse_loss_avg,
                  "train/mae_loss": mae_loss_avg,
                  "train/learning_rate": lr,
                  "epoch": epoch,
                  "iter": i
                })

                mse_losses = []
                mae_losses = []

            if (i + 1) % eval_steps == 0:
              val_loss, val_mae = evaluate(model, val_loader, device, criterion)
              wandb.log({
                "val/mse_loss": val_loss,
                "val/mae_loss": val_mae,
                "epoch": epoch
              })

        save_checkpoint(model, optimizer, epoch, len(train_loader))
    
    wandb.finish()

@torch.no_grad()
def evaluate(
    model,
    val_loader,
    device,
    criterion
):
    print("Evaluating model")
    model.eval()

    mse_losses = []
    mae_losses = []

    for i, (X, y, lengths) in enumerate(val_loader):

      if i >= 300:
        break

      if i % 15 == 0:
        print(f"Evaluating step {i}")

      X = X.to(device)
      y = y.to(device)
      lengths = lengths.cpu()

      y_pred = model(X)

      mask = (y != 0).float().unsqueeze(2)
      loss = criterion(y_pred * mask, y.unsqueeze(2)) / mask.sum()
      mae = F.l1_loss(y_pred * mask, y.unsqueeze(2) * mask, reduction='sum') / mask.sum()

      mse_losses.append(loss.item())
      mae_losses.append(mae.item())

      if i % 25 == 0:
        print(f"Evaluating step {i}")
        print(y[0][:10])
        print(y_pred[0,:, 0][:10])

    mse_loss_avg = np.mean(mse_losses)
    mae_loss_avg = np.mean(mae_losses)

    print(f"Validation loss: {mse_loss_avg} MAE: {mae_loss_avg}")
    model.train()

    return mse_loss_avg, mae_loss_avg

if __name__ == "__main__":
    train(
        model,
        train_dataloader,
        val_dataloader,
        epochs,
        eval_steps=2500,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        lr_scheduler=lr_scheduler
    )
