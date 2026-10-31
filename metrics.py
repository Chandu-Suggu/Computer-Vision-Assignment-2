"""
Evaluation Metrics for CSL7360 Assignment 2.
Implements mIoU, Boundary F1 (BF1), Q-score, and CES strictly according to specifications.
"""

import numpy as np
import cv2

def compute_boundary(mask):
    """
    Computes 4-connected boundary of object pixels.
    A boundary pixel is an object pixel with a background pixel directly above,
    below, left, or right of it. Pixels outside the image count as background.
    """
    m = mask.astype(bool)
    up = np.zeros_like(m)
    up[1:] = m[:-1]
    
    down = np.zeros_like(m)
    down[:-1] = m[1:]
    
    left = np.zeros_like(m)
    left[:, 1:] = m[:, :-1]
    
    right = np.zeros_like(m)
    right[:, :-1] = m[:, 1:]
    
    # Object pixel that touches background in 4 directions (including borders)
    boundary = m & (~up | ~down | ~left | ~right)
    return boundary

def compute_boundary_f1(pred_mask, true_mask):
    """
    Computes Boundary F1 between predicted mask and ground truth mask.
    Tolerance: 5x5 window (vectorized via cv2.filter2D for efficiency).
    """
    pb = compute_boundary(pred_mask)
    tb = compute_boundary(true_mask)
    
    n_pb = int(pb.sum())
    n_tb = int(tb.sum())
    
    # If neither mask has boundary pixels, BF1 = 1.0; if only one does, BF1 = 0.0
    if n_pb == 0 and n_tb == 0:
        return 1.0
    if n_pb == 0 or n_tb == 0:
        return 0.0
        
    kernel = np.ones((5, 5), dtype=np.uint8)
    # Check if true boundary pixel lies within 5x5 window of predicted boundary
    t_near = cv2.filter2D(tb.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT) > 0
    p_near = cv2.filter2D(pb.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT) > 0
    
    precision = float((pb & t_near).sum()) / float(n_pb)
    recall = float((tb & p_near).sum()) / float(n_tb)
    
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)

class MetricTracker:
    """
    Accumulates metrics across a dataset.
    mIoU: Sums single 2x2 confusion matrix over all images.
    BF1: Averaged per-image.
    """
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.bf1_sum = 0.0
        self.count = 0

    def update(self, pred_mask, true_mask):
        p = pred_mask.astype(np.uint8)
        t = true_mask.astype(np.uint8)
        
        self.tp += int(((p == 1) & (t == 1)).sum())
        self.tn += int(((p == 0) & (t == 0)).sum())
        self.fp += int(((p == 1) & (t == 0)).sum())
        self.fn += int(((p == 0) & (t == 1)).sum())
        
        bf1 = compute_boundary_f1(p, t)
        self.bf1_sum += bf1
        self.count += 1
        return bf1

    def compute(self):
        iou_obj = self.tp / max(self.tp + self.fp + self.fn, 1)
        iou_bg = self.tn / max(self.tn + self.fp + self.fn, 1)
        miou = (iou_obj + iou_bg) / 2.0
        
        bf1 = self.bf1_sum / max(self.count, 1)
        q = 0.70 * miou + 0.30 * bf1
        return {
            "mIoU": float(miou),
            "IoU_obj": float(iou_obj),
            "IoU_bg": float(iou_bg),
            "BF1": float(bf1),
            "Q": float(q),
            "count": self.count
        }

def compute_ces(q, t_cpu_ms, s_disk_bytes):
    """
    Computes Cost-Efficient Segmentation (CES) score:
    CES = Q * [1 - 0.20 * (T_CPU / 50ms) - 0.10 * min(1, S_disk / 4MiB)] if T_CPU <= 50ms else 0.
    4 MiB = 4,194,304 bytes.
    """
    if t_cpu_ms > 50.0:
        return 0.0
    time_penalty = 0.20 * (t_cpu_ms / 50.0)
    size_penalty = 0.10 * min(1.0, float(s_disk_bytes) / 4194304.0)
    ces = q * (1.0 - time_penalty - size_penalty)
    return max(0.0, float(ces))
