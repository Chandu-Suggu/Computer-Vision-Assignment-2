# CSL7360 Assignment 2 — Track B: Cost-Efficient Segmentation under a Compute Budget

This repository implements **Track B (Fully Convolutional Network from scratch in PyTorch)** for salient object segmentation, strictly adhering to all assignment requirements and constraints.

---

## 1. Requirements & Setup

- **Python**: 3.10+ (tested on Python 3.13)
- **PyTorch**: `>= 2.0` (pure `torch` and `torch.nn`, no `torchvision.models` or external segmentation packages used)
- **OpenCV & NumPy**: `opencv-python`, `numpy`

### Single-Thread CPU Execution Compliance
As mandated by Section 2b of the assignment:
- All thread environment variables (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`) are set to `"1"`.
- `cv2.setNumThreads(1)` and `torch.set_num_threads(1)` are called before inference.
- The timed part starts with a $256 \times 256$ RGB image in memory and ends with a $256 \times 256$ binary mask `{0, 1}` in memory using `time.process_time()`.

---

## 2. Model Architecture: DS-UNet (from scratch)

- **Architecture**: A custom, lightweight U-Net built exclusively from `torch.nn` primitives.
- **Key Building Block**: Depthwise Separable Convolutions (`3x3 Depthwise Conv -> BatchNorm -> ReLU -> 1x1 Pointwise Conv -> BatchNorm -> ReLU`).
- **Encoder**: 4 downsampling stages (128x128 -> 64x64 -> 32x32 -> 16x16) with channel dimensions `[12, 24, 48, 96]`.
- **Decoder**: Bilinear upsampling with skip connections concatenated from encoder stages (`[96+48 -> 48]`, `[48+24 -> 24]`, `[24+12 -> 12]`).
- **Head**: 1x1 convolution predicting single-channel logits, followed by bilinear upsampling of continuous logits to 256x256 and thresholding at `0.0` (sigmoid prob $\ge 0.5$) for smooth, anti-aliased boundaries.
- **Total Parameters**: **34,261** parameters.
- **Saved Model Size ($S_{\text{disk}}$)**: **~106 KB** (stored in `float16` precision, $S_{\text{disk}} \ll 4\text{ MiB}$ limit).
- **Single-Thread CPU Inference ($T_{\text{CPU}}$)**: **~15 - 19 ms** (well below the $50\text{ ms}$ ceiling).

---

## 3. Quick Start & Deliverables

### A. Run Predict Script (`predict.py`)
Run inference on any image. It loads parameters, resizes to $256 \times 256$ via `cv2.INTER_AREA`, runs 1 warm-up run, performs 20 timed runs with `time.process_time()`, saves the mask as a 0 and 255 PNG, and prints exactly two lines:
```bash
python predict.py --image path/to/photo.jpg --out mask.png
```
**Sample Output**:
```text
T_CPU_ms: 17.19
S_disk_bytes: 108899
```

### B. Evaluate Metrics & Benchmark Timing (`evaluate.py`)
Evaluate mIoU (global $2 \times 2$ confusion matrix), per-image Boundary F1 (BF1), Quality $Q = 0.70 \times \text{mIoU} + 0.30 \times \text{BF1}$, and Cost-Efficient Segmentation (CES) score:
```bash
# Evaluate on DUTS-TE
python evaluate.py --images path/to/DUTS-TE-Image --masks path/to/DUTS-TE-Mask --model track_b_model.pth

# Or benchmark single-thread CPU timing across all 5,019 DUTS-TE images:
python evaluate.py --images path/to/DUTS-TE-Image --time-only --model track_b_model.pth
```

### C. Training from Scratch (`train.py`)
Train the DS-UNet on DUTS-TR from scratch using pixel-wise Binary Cross Entropy and Soft Dice Loss:
```bash
python train.py --images path/to/DUTS-TR-Image --masks path/to/DUTS-TR-Mask --epochs 5 --batch-size 32 --lr 2e-3 --out track_b_model.pth
```

### D. Processing Captured Photos (`prepare_captured.py`)
Follow Section 1c rules for the 5 to 10 captured phone photos:
1. Place your original captured phone photos into `captured/original/` (e.g. `item1.jpg`, `item2.jpg`, ...).
2. Run the helper tool:
   ```bash
   python prepare_captured.py
   ```
   This automatically resizes each photo to $256 \times 256$ with `cv2.INTER_AREA` and saves it as `captured/images_256/<stem>.png`.
3. Save your ground truth binary masks (255 for object, 0 for background) in `captured/masks_256/<stem>.png`.
4. Re-run `python prepare_captured.py` to verify and standardize all masks.

### E. Visualizations & Overlays (`visualize.py`)
Generate side-by-side semi-transparent overlay figures for your report:
```bash
python visualize.py
```
Outputs 4-panel visual comparison figures (Input Photo, Ground Truth Overlay, Prediction Overlay, Error Map) into the `visualizations/` folder.

---

## 4. File Structure

```text
cv-2/
├── captured/
│   ├── original/         # 5-10 captured phone photos
│   ├── images_256/       # Resized photos (256x256 PNG)
│   └── masks_256/        # Ground-truth masks (256x256 PNG)
├── dataset.py            # DUTS data loader, splitting, and RAM caching
├── evaluate.py           # Evaluation script for mIoU, BF1, Q, T_CPU, and CES
├── loss.py               # Combined BCE + Soft Dice loss from scratch
├── metrics.py            # Exact mIoU, Boundary F1, and CES implementations
├── model.py              # Lightweight DS-UNet architecture from scratch
├── predict.py            # Assignment-compliant predict script
├── prepare_captured.py   # Helper tool for captured photos
├── train.py              # Training pipeline from scratch
├── visualize.py          # Semi-transparent overlay and error map generator
├── track_b_model.pth     # Saved model parameters (~106 KB)
├── REPORT.md             # Complete assignment report
└── README.md             # Project documentation
```
