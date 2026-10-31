"""
CSL7360 Assignment 2 - Visualization Tool
Generates side-by-side semi-transparent overlay figures for captured photos and failure case analysis.
"""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from dataset import read_rgb_256, read_mask_256
from predict import load_pipeline, segment

def create_overlay(rgb, mask, color=(0, 255, 0), alpha=0.45):
    """Creates a semi-transparent overlay of mask on RGB photo."""
    overlay = rgb.copy().astype(np.float32)
    colored_mask = np.zeros_like(overlay)
    colored_mask[mask > 0] = color
    
    # Blend where mask > 0
    fg = mask > 0
    overlay[fg] = (1.0 - alpha) * overlay[fg] + alpha * colored_mask[fg]
    return np.clip(overlay, 0, 255).astype(np.uint8)

def generate_comparison_figure(image_path, mask_path, model, out_path, title=None):
    """Generates a clean 4-panel visual comparison: Image | True Overlay | Pred Overlay | Error Map."""
    rgb = read_rgb_256(image_path)
    true_m = read_mask_256(mask_path)
    pred_m = segment(rgb, model)

    true_overlay = create_overlay(rgb, true_m, color=(46, 204, 113), alpha=0.45)   # emerald green
    pred_overlay = create_overlay(rgb, pred_m, color=(52, 152, 219), alpha=0.45)   # bright blue

    # Error Map: Green = TP, Red = FP (over-segmentation), Yellow/Orange = FN (missed object)
    error_map = np.zeros((256, 256, 3), dtype=np.uint8)
    tp = (pred_m == 1) & (true_m == 1)
    fp = (pred_m == 1) & (true_m == 0)
    fn = (pred_m == 0) & (true_m == 1)
    error_map[tp] = [46, 204, 113]   # Green (Correct Object)
    error_map[fp] = [231, 76, 60]    # Red (False Positive)
    error_map[fn] = [241, 196, 15]   # Yellow (False Negative)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Input Photo (256x256)")
    axes[0].axis("off")

    axes[1].imshow(true_overlay)
    axes[1].set_title("Ground Truth Overlay")
    axes[1].axis("off")

    axes[2].imshow(pred_overlay)
    axes[2].set_title("Track B Prediction Overlay")
    axes[2].axis("off")

    axes[3].imshow(error_map)
    axes[3].set_title("Error Map (G:TP, R:FP, Y:FN)")
    axes[3].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {out_path}")

def visualize_all_captured(model_path="track_b_model.pth", base_dir="captured", out_dir="visualizations"):
    model = load_pipeline(model_path)
    img_dir = os.path.join(base_dir, "images_256")
    mask_dir = os.path.join(base_dir, "masks_256")
    os.makedirs(out_dir, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not img_files:
        print(f"No processed photos found in {img_dir}. Run prepare_captured.py first.")
        return

    for ip in img_files:
        stem = os.path.splitext(os.path.basename(ip))[0]
        mp = os.path.join(mask_dir, stem + ".png")
        if not os.path.exists(mp):
            print(f"Skipping {stem}: mask not found at {mp}")
            continue
        out_fig = os.path.join(out_dir, f"{stem}_overlay.png")
        generate_comparison_figure(ip, mp, model, out_fig, title=f"Captured Photo: {stem}")

if __name__ == "__main__":
    visualize_all_captured()
