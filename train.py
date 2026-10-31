"""
CSL7360 Assignment 2 - Track B: Training Script
Trains a lightweight Fully Convolutional Network (DS-UNet) from scratch in PyTorch.
Uses BCE + Soft Dice Loss, AdamW optimizer, and Cosine Annealing learning rate schedule.
Saves model parameters in float16 to minimize disk footprint (S_disk << 4 MiB).
"""

import os
import argparse
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import get_model
from dataset import get_train_val_splits, CachedDUTSDataset, read_rgb_256, read_mask_256
from loss import BCESoftDiceLoss
from metrics import MetricTracker

def validate(model, val_pairs, target_size=128, max_val=200):
    """Evaluates validation mIoU, BF1, and Q on 256x256 masks."""
    model.eval()
    tracker = MetricTracker()
    pairs_to_eval = val_pairs[:max_val] if max_val and max_val < len(val_pairs) else val_pairs
    
    with torch.no_grad():
        for ip, mp in pairs_to_eval:
            rgb = read_rgb_256(ip)
            true_mask = read_mask_256(mp)
            
            # Predict
            small = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(small).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            logits = model(t)
            logits_np = logits[0, 0].cpu().numpy()
            
            # Bilinear upsample logits to 256x256
            logits256 = cv2.resize(logits_np, (256, 256), interpolation=cv2.INTER_LINEAR)
            pred_mask = (logits256 > 0.0).astype(np.uint8)
            
            tracker.update(pred_mask, true_mask)
            
    return tracker.compute()

def train(args):
    # Utilize available CPU cores for efficient training
    n_threads = max(1, os.cpu_count() - 1 if os.cpu_count() else 4)
    torch.set_num_threads(n_threads)
    print(f"Training using {n_threads} CPU threads...", flush=True)
    
    print(f"Loading data from {args.images} and {args.masks}...", flush=True)
    train_pairs, val_pairs = get_train_val_splits(args.images, args.masks, val_ratio=0.10, seed=42)
    
    if args.max_train and args.max_train < len(train_pairs):
        train_pairs = train_pairs[:args.max_train]
    print(f"Training on {len(train_pairs)} images, Validating on {len(val_pairs)} images.", flush=True)

    # Datasets and Loaders (Cached in RAM for zero I/O latency)
    train_dataset = CachedDUTSDataset(train_pairs, target_size=args.target_size, augment=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    # Initialize model from scratch
    print(f"Instantiating DS-UNet (base_channels={args.base_channels})...", flush=True)
    model = get_model(base_channels=args.base_channels)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model total parameters: {num_params:,}", flush=True)

    criterion = BCESoftDiceLoss(dice_weight=args.dice_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    best_q = -1.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = len(train_loader)
        t_epoch_start = time.time()

        for step, (images, masks) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (step + 1) % 25 == 0 or (step + 1) == n_batches:
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step + 1}/{n_batches}] Loss: {loss.item():.4f}", flush=True)

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        epoch_duration = time.time() - t_epoch_start
        print(f"Epoch {epoch} finished in {epoch_duration:.1f}s. Average Loss: {avg_loss:.4f}", flush=True)

        # Validation
        print("Running validation...", flush=True)
        val_res = validate(model, val_pairs, target_size=args.target_size, max_val=args.max_val)
        print(f"Val mIoU: {val_res['mIoU']:.4f} | Val BF1: {val_res['BF1']:.4f} | Val Q: {val_res['Q']:.4f}", flush=True)

        # Save best model checkpoint
        if val_res['Q'] > best_q:
            best_q = val_res['Q']
            print(f"--> Validation Q improved to {best_q:.4f}! Saving checkpoint to {args.out}...", flush=True)
            
            # Save state_dict in float16 for minimal disk footprint
            fp16_state_dict = {
                k: v.half() if v.is_floating_point() else v 
                for k, v in model.state_dict().items()
            }
            checkpoint = {
                "epoch": epoch,
                "base_channels": args.base_channels,
                "target_size": args.target_size,
                "val_q": best_q,
                "val_metrics": val_res,
                "state_dict": fp16_state_dict
            }
            torch.save(checkpoint, args.out)
            s_bytes = os.path.getsize(args.out)
            print(f"--> Saved {args.out} ({s_bytes:,} bytes, {s_bytes / 1024:.1f} KB)", flush=True)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.1f} minutes. Best Validation Q: {best_q:.4f}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Train Track B DS-UNet on DUTS-TR")
    parser.add_argument("--images", default=r"c:\Downloads\Acads\CV\Assignment_2\Track_A_Computer_Vision_Assignment\DUTS-TR\DUTS-TR-Image")
    parser.add_argument("--masks", default=r"c:\Downloads\Acads\CV\Assignment_2\Track_A_Computer_Vision_Assignment\DUTS-TR\DUTS-TR-Mask")
    parser.add_argument("--out", default="track_b_model.pth", help="Output path for best checkpoint")
    parser.add_argument("--base-channels", type=int, default=12, help="Base channel multiplier (12 or 16)")
    parser.add_argument("--target-size", type=int, default=128, help="Internal resolution (128)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--dice-weight", type=float, default=1.0, help="Weight of soft dice loss")
    parser.add_argument("--max-train", type=int, default=0, help="Limit number of training samples (0 for all)")
    parser.add_argument("--max-val", type=int, default=250, help="Number of validation images per epoch")
    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
