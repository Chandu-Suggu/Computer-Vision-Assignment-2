"""
CSL7360 Assignment 2 - Track B: Evaluation and Benchmark Timing Script
Computes mIoU, Boundary F1, Q score, T_CPU, and CES score on DUTS-TE or any validation/test set.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import glob
import time
import cv2
import numpy as np
import torch

cv2.setNumThreads(1)
torch.set_num_threads(1)

from dataset import read_rgb_256, read_mask_256, load_pairs, get_train_val_splits
from predict import load_pipeline, segment
from metrics import MetricTracker, compute_ces

def evaluate_dataset(pairs, model, verbose=True):
    """Evaluates mIoU, BF1, and Q score across the provided image-mask pairs."""
    tracker = MetricTracker()
    total = len(pairs)
    
    for i, (ip, mp) in enumerate(pairs):
        rgb = read_rgb_256(ip)
        true_mask = read_mask_256(mp)
        pred_mask = segment(rgb, model)
        tracker.update(pred_mask, true_mask)

        if verbose and ((i + 1) % 500 == 0 or (i + 1) == total):
            print(f"Evaluated {i + 1}/{total} images...")

    res = tracker.compute()
    if verbose:
        print("----------------------------------------")
        print(f"Total Images: {res['count']}")
        print(f"mIoU:        {res['mIoU']:.5f} (Object: {res['IoU_obj']:.5f}, Background: {res['IoU_bg']:.5f})")
        print(f"Boundary F1: {res['BF1']:.5f}")
        print(f"Q Score:     {res['Q']:.5f}")
        print("----------------------------------------")
    return res

def benchmark_timing(image_dir, model, model_path, verbose=True):
    """
    Measures average single-thread CPU process_time over all images in image_dir.
    Computes CES score.
    """
    img_files = sorted(glob.glob(os.path.join(image_dir, "*")))
    if not img_files:
        raise RuntimeError(f"No images found in {image_dir}")
    
    n_images = len(img_files)
    s_disk_bytes = os.path.getsize(model_path)
    
    # 1. Warm-up run on first photo (Section 2b: first run is slow while libraries load)
    first_rgb = read_rgb_256(img_files[0])
    _ = segment(first_rgb, model)

    # 2. Timing loop with time.process_time()
    total_cpu_time = 0.0
    for i, fpath in enumerate(img_files):
        rgb = read_rgb_256(fpath)
        t0 = time.process_time()
        _ = segment(rgb, model)
        total_cpu_time += (time.process_time() - t0)

        if verbose and ((i + 1) % 500 == 0 or (i + 1) == n_images):
            print(f"Timed {i + 1}/{n_images} images...")

    t_cpu_ms = (total_cpu_time / n_images) * 1000.0
    print("================ TIMING RESULTS ================")
    print(f"Images evaluated: {n_images}")
    print(f"T_CPU (avg ms):   {t_cpu_ms:.2f} ms")
    print(f"S_disk (bytes):   {s_disk_bytes} bytes ({s_disk_bytes / 1024:.1f} KB)")
    print(f"Budget <= 50 ms:  {'PASS' if t_cpu_ms <= 50.0 else 'FAIL'}")
    print("================================================")
    return t_cpu_ms, s_disk_bytes

def main():
    parser = argparse.ArgumentParser(description="Evaluate Track B Model")
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--masks", default=None, help="Path to masks directory (for accuracy evaluation)")
    parser.add_argument("--model", default="track_b_model.pth", help="Path to saved model parameters")
    parser.add_argument("--time-only", action="store_true", help="Only run CPU timing benchmark")
    parser.add_argument("--val-split", action="store_true", help="Evaluate on DUTS-TR validation split (10%)")
    args = parser.parse_args()

    print(f"Loading pipeline from {args.model}...")
    model = load_pipeline(args.model)

    if args.val_split:
        if not args.masks:
            raise ValueError("--masks must be provided when using --val-split")
        print("Extracting DUTS-TR validation split (10%)...")
        _, val_pairs = get_train_val_splits(args.images, args.masks)
        print(f"Evaluating {len(val_pairs)} validation images...")
        res = evaluate_dataset(val_pairs, model)
        return

    if args.time_only:
        benchmark_timing(args.images, model, args.model)
        return

    if args.masks is not None:
        pairs = load_pairs(args.images, args.masks)
        print(f"Evaluating accuracy on {len(pairs)} pairs...")
        res = evaluate_dataset(pairs, model)
        
        # Also compute timing on the test set if >= 100 images
        if len(pairs) >= 100:
            print("Measuring average CPU timing...")
            t_cpu_ms, s_disk_bytes = benchmark_timing(args.images, model, args.model, verbose=False)
            ces = compute_ces(res["Q"], t_cpu_ms, s_disk_bytes)
            print("================ FINAL CES ================")
            print(f"Quality Q:    {res['Q']:.5f}")
            print(f"T_CPU (ms):   {t_cpu_ms:.2f}")
            print(f"S_disk (B):   {s_disk_bytes}")
            print(f"CES Score:    {ces:.5f}")
            print("===========================================")
    else:
        print("No masks provided, running timing benchmark...")
        benchmark_timing(args.images, model, args.model)

if __name__ == "__main__":
    main()
