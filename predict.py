"""
CSL7360 Assignment 2 - Track B: Predict script
Command: python predict.py --image photo.jpg --out mask.png

Strict compliance with assignment specifications:
- Sets single-thread CPU environment variables before imports.
- Resizes input photo to 256x256 using cv2.INTER_AREA.
- Segments image, saves mask as 0 and 255 PNG.
- Measures 20 timed runs using time.process_time().
- Outputs exactly two lines:
    T_CPU_ms: <float>
    S_disk_bytes: <int>
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import time
import cv2
import numpy as np
import torch

cv2.setNumThreads(1)
torch.set_num_threads(1)

from model import get_model

SIZE = 256
NET_SIZE = 128

def load_pipeline(model_path="track_b_model.pth"):
    """Loads model checkpoint and prepares optimized TorchScript CPU inference model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Parameter file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    base_channels = checkpoint.get("base_channels", 12)
    state_dict = checkpoint["state_dict"]
    
    # Cast weights back to float32 for CPU computation if stored as float16
    float_state_dict = {k: v.float() if v.is_floating_point() else v for k, v in state_dict.items()}
    
    model = get_model(base_channels=base_channels)
    model.load_state_dict(float_state_dict)
    model.eval()

    # Create TorchScript model for optimal single-thread CPU performance
    dummy = torch.randn(1, 3, NET_SIZE, NET_SIZE)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)
        traced = torch.jit.optimize_for_inference(traced)
    return traced

def segment(rgb256, model):
    """
    Timed segmentation pipeline:
    Starts with 256x256 photo in memory, ends with 256x256 binary mask {0, 1} in memory.
    All steps in between count.
    """
    # 1. Downscale to working resolution
    small = cv2.resize(rgb256, (NET_SIZE, NET_SIZE), interpolation=cv2.INTER_AREA)
    # 2. Normalize and convert to float tensor [1, 3, H, W]
    t = torch.from_numpy(small).permute(2, 0, 1).unsqueeze(0).float().mul_(1.0 / 255.0)
    # 3. Network inference
    logits = model(t)
    # 4. Bilinear upsampling of continuous logits to 256x256 gives clean sub-pixel edges
    logits_np = logits[0, 0].numpy()
    logits256 = cv2.resize(logits_np, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    # 5. Threshold at 0 (sigmoid prob >= 0.5) to produce final 256x256 mask {0, 1}
    mask256 = (logits256 > 0.0).astype(np.uint8)
    return mask256

def main():
    parser = argparse.ArgumentParser(description="Track B: FCN Salient Object Segmentation Predictor")
    parser.add_argument("--image", required=True, help="Path to input photo")
    parser.add_argument("--out", required=True, help="Path to save output mask PNG")
    parser.add_argument("--model", default="track_b_model.pth", help="Path to saved model weights")
    args = parser.parse_args()

    s_disk_bytes = os.path.getsize(args.model)
    model = load_pipeline(args.model)

    # Read original photo and resize to 256x256 using cv2.INTER_AREA (Section 1b)
    im = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"Could not read image from {args.image}")
    rgb256 = cv2.cvtColor(cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

    # First run (warmup + save output mask)
    pred_mask = segment(rgb256, model)
    cv2.imwrite(args.out, (pred_mask * 255).astype(np.uint8))

    # 20 timed runs using time.process_time()
    ts = []
    for _ in range(20):
        t0 = time.process_time()
        _ = segment(rgb256, model)
        ts.append(time.process_time() - t0)

    t_cpu_ms = float(np.mean(ts)) * 1000.0

    # Print EXACTLY two lines as required by assignment specification
    print(f"T_CPU_ms: {t_cpu_ms:.2f}")
    print(f"S_disk_bytes: {s_disk_bytes}")

if __name__ == "__main__":
    main()
