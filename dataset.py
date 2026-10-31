"""
Dataset handling and loading for DUTS and captured photos.
Implements Section 1b resizing rules:
- Photos resized to 256x256 with cv2.INTER_AREA.
- Masks resized to 256x256 with cv2.INTER_NEAREST.
- Mask values >= 128 become 1, others 0.
Includes an in-memory cached dataset for ultra-fast training without disk bottlenecks.
"""

import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

SIZE = 256

def read_rgb_256(path):
    """Reads image, resizes to 256x256 using cv2.INTER_AREA, returns RGB uint8 array."""
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"Cannot read image: {path}")
    im = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def read_mask_256(path):
    """Reads mask, resizes to 256x256 using cv2.INTER_NEAREST, returns {0, 1} uint8 array."""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError(f"Cannot read mask: {path}")
    m = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return (m >= 128).astype(np.uint8)

def load_pairs(image_dir, mask_dir):
    """Matches images and masks by base filename stem."""
    im_files = sorted(glob.glob(os.path.join(image_dir, "*")))
    mask_files = {}
    for p in glob.glob(os.path.join(mask_dir, "*")):
        stem = os.path.splitext(os.path.basename(p))[0]
        mask_files[stem] = p

    pairs = []
    for ip in im_files:
        stem = os.path.splitext(os.path.basename(ip))[0]
        if stem in mask_files:
            pairs.append((ip, mask_files[stem]))

    if not pairs:
        raise RuntimeError(f"No matching pairs found between {image_dir} and {mask_dir}")
    return pairs

def get_train_val_splits(image_dir, mask_dir, val_ratio=0.10, seed=42):
    """
    Splits DUTS-TR into deterministic train and validation splits.
    Default: 90% train (approx 9,498), 10% val (approx 1,055).
    """
    pairs = load_pairs(image_dir, mask_dir)
    rng = random.Random(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)
    
    val_size = int(len(shuffled) * val_ratio)
    val_pairs = shuffled[:val_size]
    train_pairs = shuffled[val_size:]
    return train_pairs, val_pairs

class CachedDUTSDataset(Dataset):
    """
    In-memory cached dataset for ultra-fast training without disk bottlenecks.
    Pre-resizes images and masks to target_size upon creation and stores them in RAM.
    """
    def __init__(self, pairs, target_size=128, augment=False, max_samples=None):
        self.pairs = pairs[:max_samples] if max_samples else pairs
        self.target_size = target_size
        self.augment = augment
        
        self.images = []
        self.masks = []
        
        print(f"Pre-caching {len(self.pairs)} samples into RAM at {target_size}x{target_size}...", flush=True)
        for i, (ip, mp) in enumerate(self.pairs):
            im = cv2.imread(ip, cv2.IMREAD_COLOR)
            if im is None:
                continue
            im = cv2.resize(im, (target_size, target_size), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            m = cv2.resize(m, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
            bin_m = (m >= 128).astype(np.uint8)
            
            self.images.append(rgb)
            self.masks.append(bin_m)
            
            if (i + 1) % 500 == 0 or (i + 1) == len(self.pairs):
                print(f"  Cached {i + 1}/{len(self.pairs)} images...", flush=True)
                
        self.length = len(self.images)
        print(f"Done. Successfully cached {self.length} samples in RAM.", flush=True)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rgb = self.images[idx]
        mask = self.masks[idx]

        if self.augment:
            if random.random() > 0.5:
                rgb = np.fliplr(rgb).copy()
                mask = np.fliplr(mask).copy()
            if random.random() > 0.5:
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-12, 12)
                rgb = np.clip(alpha * rgb + beta, 0, 255).astype(np.uint8)

        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return img_t, mask_t
