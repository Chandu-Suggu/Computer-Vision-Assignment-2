"""
CSL7360 Assignment 2 - Captured Photos Processing Tool
Handles Section 1c rules:
- Resizes original captured photos to 256x256 PNGs using cv2.INTER_AREA into captured/images_256/
- Standardizes user masks into 256x256 binary PNGs (0 and 255) using cv2.INTER_NEAREST into captured/masks_256/
- Validates matching filename stems across original/, images_256/, and masks_256/
"""

import os
import glob
import argparse
import cv2
import numpy as np

SIZE = 256

def process_captured_photos(base_dir="captured"):
    orig_dir = os.path.join(base_dir, "original")
    img_dir = os.path.join(base_dir, "images_256")
    mask_dir = os.path.join(base_dir, "masks_256")

    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    orig_files = sorted(glob.glob(os.path.join(orig_dir, "*")))
    # Filter image extensions
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    orig_files = [f for f in orig_files if os.path.splitext(f.lower())[1] in valid_exts]

    if not orig_files:
        print(f"No photos found in {orig_dir}!")
        print("Please place 5 to 10 photos of everyday objects in captured/original/ (e.g. item1.jpg, item2.jpg, ...)")
        return

    print(f"Found {len(orig_files)} original photo(s) in {orig_dir}:")
    for f in orig_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        # 1. Resize original photo to 256x256 with cv2.INTER_AREA
        im = cv2.imread(f, cv2.IMREAD_COLOR)
        if im is None:
            print(f"  [Warning] Could not read {f}, skipping.")
            continue
        im_256 = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        out_img_path = os.path.join(img_dir, stem + ".png")
        cv2.imwrite(out_img_path, im_256)
        print(f"  -> Resized and saved {out_img_path}")

        # 2. Check mask
        mask_candidates = [
            os.path.join(mask_dir, stem + ".png"),
            os.path.join(mask_dir, stem + ".jpg"),
            os.path.join(orig_dir, stem + "_mask.png"),
            os.path.join(orig_dir, stem + "_mask.jpg"),
        ]
        found_mask = None
        for mc in mask_candidates:
            if os.path.exists(mc):
                found_mask = mc
                break

        final_mask_path = os.path.join(mask_dir, stem + ".png")
        if found_mask:
            m = cv2.imread(found_mask, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m_256 = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
                # Binarize with threshold 128: 0 for background, 255 for object
                bin_mask = np.where(m_256 >= 128, 255, 0).astype(np.uint8)
                cv2.imwrite(final_mask_path, bin_mask)
                print(f"  -> Verified and normalized mask {final_mask_path}")
        else:
            if not os.path.exists(final_mask_path):
                print(f"  [Notice] Mask not yet created for {stem}. Draw mask and save to: {final_mask_path}")

    # Summary report
    print("\n--- Captured Photos Status ---")
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    img_stems = {os.path.splitext(os.path.basename(p))[0] for p in imgs}
    mask_stems = {os.path.splitext(os.path.basename(p))[0] for p in masks}
    matched = img_stems.intersection(mask_stems)

    print(f"Total processed images: {len(imgs)}")
    print(f"Total valid masks:      {len(masks)}")
    print(f"Fully matched pairs:    {len(matched)}")
    for stem in sorted(img_stems):
        status = "READY" if stem in mask_stems else "MISSING MASK"
        print(f"  - {stem}: {status}")

if __name__ == "__main__":
    process_captured_photos()
