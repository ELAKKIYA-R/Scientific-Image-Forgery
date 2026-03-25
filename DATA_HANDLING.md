# Data Handling Strategy

## Overview

This document explains how the dataset handles authentic vs forged images, especially when they share the same filename.

## Key Points

### 1. Image Naming Conflict
- **Problem**: Authentic and forged images can have the same filename (e.g., `12345.png` exists in both `authentic/` and `forged/` folders)
- **Solution**: The dataset determines authentic vs forged by checking if a corresponding mask file exists

### 2. Mask Loading Logic

The `ForgeryDetectionDataset` uses the following logic:

1. **If label is explicitly 0 (authentic)**:
   - Always return zero mask, regardless of whether a mask file exists
   - This ensures authentic images never have forgery masks

2. **If label is 1 (forged) or unknown**:
   - Check if mask file exists in `train_masks/` or `supplemental_masks/`
   - If mask exists → load it (forged image)
   - If mask doesn't exist → return zero mask (authentic image)

3. **Label determination** (in `_load_clustered_split`):
   - Check if `{image_id}.npy` exists in mask directories
   - If exists → label = 1 (forged)
   - If not exists → label = 0 (authentic)

### 3. Clustering Script Behavior

The `cluster_image.py` script:
- Clusters **both** authentic and forged images together
- Copies images to cluster folders in `split_data/`
- **Important**: If both authentic and forged versions of the same image exist, the later copy overwrites the earlier one
- The dataset will correctly identify which is which by checking mask existence

### 4. Training Data

- **Authentic images**: Get zero masks (no forgery regions)
- **Forged images**: Load masks from `.npy` files (1-3 channels)
- Both types are used in training to learn the distinction

## Verification

To verify the logic is working correctly:

```python
from dataset import create_dataloaders
from pathlib import Path

train_loader, val_loader = create_dataloaders(
    data_root=Path('recodai-luc-scientific-image-forgery-detection'),
    batch_size=4
)

# Check that authentic images have zero masks
for batch in train_loader:
    for i, label in enumerate(batch['label']):
        if label == 0:  # Authentic
            assert batch['mask'][i].max() == 0, "Authentic image should have zero mask!"
```

## Summary

✅ **Authentic images** (label=0): Always get zero masks  
✅ **Forged images** (label=1): Load masks from `.npy` files  
✅ **Mask existence** determines the label when using clustered splits  
✅ **Both image types** are used in training for balanced learning

