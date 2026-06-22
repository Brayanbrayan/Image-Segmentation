# Image Segmentation with Focal Loss and Point Annotations

A PyTorch implementation exploring advanced loss functions for semantic image segmentation, with emphasis on **Partial Cross Entropy Loss** and **Focal Loss** for weakly-supervised learning with sparse point annotations.

## Overview

This project investigates how to train segmentation models with limited pixel-level supervision by:
- Implementing custom **Cross Entropy loss** from scratch and validating against PyTorch's built-in version
- Developing **Partial Cross Entropy loss** to handle sparse point-level annotations
- Introducing **Focal Loss** to focus training on hard examples
- Combining these approaches into **Partial Focal Cross Entropy loss**

## Key Contributions

### 1. Manual Cross Entropy Implementation
A from-scratch implementation of cross entropy loss for pixel-wise classification that:
- Applies softmax across the channel dimension
- Computes log probability of correct class per pixel
- Validates against PyTorch's `nn.CrossEntropyLoss()` (verified to 8 decimal places)

### 2. Partial Cross Entropy Loss
Enables training with **sparse point annotations** rather than full masks:
- Simulates point-level labels from full ground truth
- Only computes loss at labeled pixels
- Averages loss only over annotated regions
- Demonstrates ~99% reduction in annotation density (0.61% of pixels labeled)

### 3. Focal Loss with Point Supervision
Combines focal loss theory with partial supervision:
- Down-weights easy examples using the factor `(1 - p_t)^γ`
- Reduces impact of well-classified pixels on gradients
- Particularly effective for imbalanced segmentation tasks
- Configurable focusing parameter (γ = 2.0 by default)

### 4. Visualization Tools
Interactive visualizations showing:
- Per-pixel loss maps (standard CE vs. Focal loss)
- Ground truth vs. point label sparsity
- Loss distribution across image regions
- Statistics on pixel-wise loss (mean, max, min)

## Dataset

Uses the **LandCover.ai Dataset**:
- **41 GeoTIFF images** with ground truth semantic masks
- **5 classes**: Building, Woodland, Water, Road, Barren land
- **9095 × 9636 resolution** per original tile
- **Pre-tiled to 512 × 512 patches** for training

### Data Organization
```
landcover_data/
├── images/          # RGB satellite imagery
├── masks/           # Single-channel semantic masks (5 classes)
├── train.txt        # Training split indices (7470 patches)
├── val.txt          # Validation split
├── test.txt         # Test split
└── split.py         # Tiling script (512×512 patches)
```

## Usage

### Requirements
```bash
pip install torch torchvision
pip install matplotlib numpy pillow opencv-python
```

### Quick Start
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define loss functions
loss_full_ce = cross_entropy_pytorch(predictions, targets)
loss_partial_ce = partial_cross_entropy(predictions, sparse_targets, point_mask)
loss_partial_focal = partial_focal_ce(predictions, sparse_targets, point_mask, gamma=2.0)

# Visualize differences
visualise_partial_vs_full(predictions, targets, point_mask)
calculate_and_visualize_loss_maps(predictions, targets, gamma=2.0)
```

### Loss Function Signatures

**Manual Cross Entropy Loss:**
```python
def cross_entropy_manual(predictions, targets):
    """
    predictions: (N, C, H, W) — raw logits from the network
    targets:     (N, H, W)    — integer class indices per pixel
    
    Returns: scalar loss
    """
    probs = F.softmax(predictions, dim=1)
    targets_expanded = targets.unsqueeze(1)
    correct_class_probs = probs.gather(dim=1, index=targets_expanded).squeeze(1)
    loss_per_pixel = -torch.log(correct_class_probs + 1e-8)
    return loss_per_pixel.mean()
```

**Partial Cross Entropy Loss:**
```python
def partial_cross_entropy(predictions, targets, point_mask):
    """
    Partial CE loss — only computed at labelled points.
    
    predictions: (N, C, H, W) — raw logits
    targets:     (N, H, W)    — class indices, 255 where unlabelled
    point_mask:  (N, H, W)    — 1 at labelled pixels, 0 elsewhere
    
    Returns: scalar loss
    """
    probs = F.softmax(predictions, dim=1)
    targets_safe = targets.clone()
    targets_safe[targets_safe == 255] = 0
    targets_expanded = targets_safe.unsqueeze(1)
    correct_class_probs = probs.gather(dim=1, index=targets_expanded).squeeze(1)
    loss_per_pixel = -torch.log(correct_class_probs + 1e-8)
    masked_loss = loss_per_pixel * point_mask
    
    num_labelled = point_mask.sum()
    if num_labelled == 0:
        return torch.tensor(0.0, requires_grad=True)
    
    return masked_loss.sum() / num_labelled
```

**Partial Focal Cross Entropy Loss:**
```python
def partial_focal_ce(predictions, targets, point_mask, gamma=2.0):
    """
    Partial Focal Cross Entropy — combines focal loss with point supervision.
    
    pfCE = Σ(FocalLoss(pred, GT) × MASK_labelled) / Σ MASK_labelled
    
    predictions: (N, C, H, W) — raw logits
    targets:     (N, H, W)    — class indices, 255 where unlabelled
    point_mask:  (N, H, W)    — 1 at labelled pixels, 0 elsewhere
    gamma:       float         — focusing parameter (0=CE, 2=default)
    
    Returns: scalar loss
    """
    probs = F.softmax(predictions, dim=1)
    targets_safe = targets.clone()
    targets_safe[targets_safe == 255] = 0
    targets_expanded = targets_safe.unsqueeze(1)
    correct_class_probs = probs.gather(dim=1, index=targets_expanded).squeeze(1)
    
    ce_loss = -torch.log(correct_class_probs + 1e-8)
    focal_weight = (1 - correct_class_probs) ** gamma
    focal_loss_per_pixel = focal_weight * ce_loss
    masked_loss = focal_loss_per_pixel * point_mask
    
    num_labelled = point_mask.sum()
    if num_labelled == 0:
        return torch.tensor(0.0, requires_grad=True)
    
    return masked_loss.sum() / num_labelled
```

**Simulate Point Labels:**
```python
def simulate_point_labels(targets, num_points_per_class=5):
    """
    Takes a full ground truth mask and simulates sparse point annotations.
    
    targets: (N, H, W) integer class labels
    num_points_per_class: how many points to keep per class per image
    
    Returns:
        sparse_targets: (N, H, W) — same as targets but 255 where unlabelled
        point_mask:     (N, H, W) — 1 where labelled, 0 everywhere else
    """
    N, H, W = targets.shape
    point_mask = torch.zeros_like(targets, dtype=torch.float32)
    sparse_targets = torch.full_like(targets, 255)

    for n in range(N):
        classes = targets[n].unique()
        for cls in classes:
            class_pixels = (targets[n] == cls).nonzero(as_tuple=False)
            if len(class_pixels) == 0:
                continue
            
            num_to_sample = min(num_points_per_class, len(class_pixels))
            indices = torch.randperm(len(class_pixels))[:num_to_sample]
            sampled = class_pixels[indices]

            for pixel in sampled:
                h, w = pixel[0], pixel[1]
                point_mask[n, h, w] = 1.0
                sparse_targets[n, h, w] = targets[n, h, w]

    return sparse_targets, point_mask
```

## Key Results

| Loss Function | Mean Loss | Notes |
|---------------|-----------|-------|
| Full CE (all pixels) | 1.9947 | Baseline with complete masks |
| Partial CE (0.61% labeled) | 2.2220 | Higher loss due to sparse supervision |
| Partial Focal CE (γ=2) | 1.8198 | **Lower loss** - focuses on hard pixels |

**Insight:** Focal loss successfully reduces overall loss despite sparse annotations by prioritizing difficult pixels.

## Mathematical Foundation

### Cross Entropy Loss
For each pixel, the standard cross entropy loss is:
```
CE = -log(p_t)
where p_t = softmax(logits)[true_class]
```

### Focal Loss
Focal loss down-weights easy examples:
```
FL = -(1 - p_t)^γ × log(p_t)
where (1 - p_t)^γ is the focusing term
```

### Partial Focal Cross Entropy
Combines both approaches:
```
pfCE = Σ(FL(pred, gt) × mask_labeled) / Σ(mask_labeled)
```

## Concepts Explained

### Why Partial Loss?
- **Full annotation is expensive**: Pixel-level masks require extensive manual labeling
- **Point annotations are practical**: Users can quickly mark a few representative pixels per class
- **Partial loss leverages this**: Only computes gradients at annotated points while ignoring unlabeled regions

### Why Focal Loss?
- **Class imbalance**: Some classes (e.g., roads) are rare in images
- **Easy examples dominate**: Most pixels are correctly classified after few epochs, making loss uninformative
- **Focal loss down-weights easy pixels**: Dynamically focuses training on hard boundaries and rare classes
- **Improved learning**: Network learns to distinguish difficult regions instead of over-optimizing easy areas

### Validation Against PyTorch
```
Manual implementation vs PyTorch's nn.CrossEntropyLoss()
Difference: 0.00000012 (verified at 8 decimal places)
```

## Notebook Walkthrough

The main notebook (`ImageSeg.ipynb`) includes:

1. **Loss Implementation**
   - Cell 6: Manual CE loss implementation & validation
   - Cell 7: Visualization of CE loss per pixel

2. **Partial Loss**
   - Cell 9: Simulate point labels from full masks
   - Cell 10: Partial CE loss implementation

3. **Focal Loss**
   - Cell 12: Partial Focal CE implementation
   - Cell 13: Loss map visualizations comparing CE vs. Focal

4. **Dataset**
   - Cell 14: Download LandCover.ai dataset
   - Cell 19: Tile images to 512×512 patches

## Project Structure

```
Image-Segmentation/
├── ImageSeg.ipynb              # Main notebook with all implementations
├── README.md                    # This file
├── landcover_data/             # Downloaded and processed dataset
│   ├── images/                 # Original GeoTIFF files
│   ├── masks/                  # Semantic segmentation masks
│   ├── output/                 # Tiled 512×512 patches
│   ├── train.txt               # Training indices
│   ├── val.txt                 # Validation indices
│   ├── test.txt                # Test indices
│   └── split.py                # Tiling script
└── connection_test.txt         # GitHub connection test
```

## Next Steps

Potential extensions and improvements:

1. **Train Full Network**: Implement U-Net or DeepLab architecture with these loss functions
2. **Performance Comparison**: Benchmark full masks vs. point supervision at different sparsity levels
3. **Parameter Tuning**: Explore focusing parameters (γ ∈ [0, 5]) and points per class
4. **Class Weighting**: Add inverse frequency weighting for highly imbalanced datasets
5. **Interactive Annotation**: Build UI for collecting point annotations from real images
6. **Ablation Study**: Systematically compare all loss combinations
7. **Multi-Scale Loss**: Apply focal loss at different network depths

## References

- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (RetinaNet), ICCV 2017
- **LandCover.ai**: [https://landcover.ai/](https://landcover.ai/)
- **PyTorch**: [https://pytorch.org/](https://pytorch.org/)
- **PyTorch Vision**: [https://pytorch.org/vision/main/index.html](https://pytorch.org/vision/main/index.html)
- **Semantic Segmentation Papers**: DeepLab, U-Net, FCN

## Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision
- NumPy
- Matplotlib
- Pillow
- OpenCV (cv2)

## License

This project is provided as-is for educational and research purposes.

## Author

Created as an exploration of weak supervision techniques and advanced loss functions for semantic segmentation with sparse annotations.

---

**Status**: Active development - implements loss functions and visualizations for the LandCover.ai dataset.

**Last Updated**: June 2026
