"""
Dataset for Scientific Image Forgery Detection

Handles:
- Multi-channel masks (1, 2, or 3 channels)
- RGB and RGBA images  
- Authentic images (zero masks)
- Train/val split from clustering
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class ForgeryDetectionDataset(Dataset):
    """
    Dataset for copy-move forgery detection.
    
    Handles both forged images (with masks) and authentic images (zero masks).
    
    Mask format:
    - Forged images have .npy masks with 1-3 channels (representing different regions)
    - Authentic images get all-zero masks
    - Option to combine channels or keep separate:
      * combine_channels=False: Keep multi-channel (helps learn source vs target regions)
      * combine_channels=True: Max across channels to single channel (simpler)
    
    Image format:
    - RGBA images are converted to RGB (alpha composited on white)
    - All images normalized to [0, 1] then standardized
    """
    
    def __init__(
        self,
        image_paths: List[Path],
        mask_dir: Optional[Union[Path, 'CombinedMaskDir']] = None,
        labels: Optional[List[int]] = None,  # 0=authentic, 1=forged
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (512, 512),
        num_mask_channels: int = 1,  # Always single channel now
        combine_mask_channels: bool = True,  # Always combine to single channel
        is_test: bool = False
    ):
        """
        Args:
            image_paths: List of image file paths
            mask_dir: Directory containing .npy masks (None for test), or CombinedMaskDir
            labels: List of labels (0=authentic, 1=forged)
            transform: Albumentations transform
            image_size: Target (H, W) for resizing
            num_mask_channels: Number of output mask channels (default 1 for single channel)
            combine_mask_channels: If True, max across channels to get single channel (always True now)
            is_test: Whether this is test set (no masks/labels)
        """
        self.image_paths = image_paths
        # Handle both Path and CombinedMaskDir
        if isinstance(mask_dir, CombinedMaskDir):
            self.mask_dir = mask_dir
        elif mask_dir is not None:
            self.mask_dir = Path(mask_dir)
        else:
            self.mask_dir = None
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        self.combine_mask_channels = combine_mask_channels
        # Always use single channel (masks are converted to single channel in _load_mask)
        self.num_mask_channels = 1
        self.is_test = is_test
        
        # Create image_id to label mapping if labels provided
        if labels is not None:
            assert len(labels) == len(image_paths)
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - 'image': Tensor (3, H, W)
                - 'mask': Tensor (num_mask_channels, H, W) 
                - 'label': Tensor (1,) - 0=authentic, 1=forged
                - 'valid_mask_channels': Tensor (num_mask_channels,) - which channels are valid
                - 'image_id': str - image filename without extension
        """
        img_path = Path(self.image_paths[idx])
        image_id = img_path.stem
        
        # Load image
        image = self._load_image(img_path)
        
        # Determine label: filename-based is most reliable
        # In dataset_split structure: "authentic_12345.png" or "forged_12345.png"
        if image_id.startswith('authentic_'):
            label = 0  # Authentic
        elif image_id.startswith('forged_'):
            label = 1  # Forged
        elif self.labels is not None:
            # Fallback to provided labels if filename doesn't indicate type
            label = self.labels[idx]
        elif self.is_test:
            label = -1  # Unknown for test
        else:
            # Last resort: infer from mask existence (will be refined after loading)
            label = None  # Will infer after mask load
        
        # Load or create mask (pass label to ensure authentic images get zero masks)
        mask, valid_channels = self._load_mask(image_id, image.shape[:2], label=label)
        
        # If label still wasn't set, infer from mask
        if label is None:
            # If mask has any non-zero values, it's forged; otherwise authentic
            label = 1 if mask.max() > 0 else 0
        
        # Apply transforms
        if self.transform:
            # Handle single channel mask (or multi-channel if needed)
            # Split mask into individual channels for albumentations
            masks_dict = {}
            for i in range(mask.shape[2]):
                masks_dict[f'mask{i}'] = mask[:, :, i]
            
            # Create additional_targets for all mask channels
            additional_targets = {f'mask{i}': 'mask' for i in range(mask.shape[2])}
            
            # Apply transform with additional targets
            transform_with_masks = A.Compose(
                self.transform.transforms,
                additional_targets=additional_targets
            )
            
            transformed = transform_with_masks(image=image, **masks_dict)
            image = transformed['image']
            
            # Reconstruct mask (single channel now, but code handles both)
            mask_channels = []
            for i in range(mask.shape[2]):
                m = transformed[f'mask{i}']
                if isinstance(m, torch.Tensor):
                    mask_channels.append(m.unsqueeze(0) if m.dim() == 2 else m)
                else:
                    mask_channels.append(torch.from_numpy(m).unsqueeze(0).float())
            mask = torch.cat(mask_channels, dim=0)
        else:
            # Default: just resize and convert
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            mask = cv2.resize(
                mask, (self.image_size[1], self.image_size[0]),
                interpolation=cv2.INTER_NEAREST
            )
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        
        # Ensure mask has correct number of channels and shape
        if isinstance(mask, torch.Tensor):
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            # Ensure mask is (C, H, W)
            if mask.shape[0] > mask.shape[2] and len(mask.shape) == 3:
                # Mask might be in (H, W, C) format, transpose it
                mask = mask.permute(2, 0, 1)
            
            # Masks are already converted to single channel in _load_mask
            # But ensure single channel here as well (redundant but safe)
            if mask.shape[0] > 1:
                # Take max across channels to get single binary mask
                mask = mask.max(dim=0, keepdim=True)[0]
            
            # Ensure correct number of channels
            if mask.shape[0] < self.num_mask_channels:
                padding = torch.zeros(
                    self.num_mask_channels - mask.shape[0],
                    mask.shape[1], mask.shape[2],
                    dtype=mask.dtype, device=mask.device
                )
                mask = torch.cat([mask, padding], dim=0)
            elif mask.shape[0] > self.num_mask_channels:
                mask = mask[:self.num_mask_channels]
        
        return {
            'image': image,
            'mask': mask,
            'label': torch.tensor(label, dtype=torch.long),
            'valid_mask_channels': torch.tensor(valid_channels, dtype=torch.float32),
            'image_id': image_id
        }
    
    def _load_image(self, path: Path) -> np.ndarray:
        """
        Load image and convert to RGB.
        
        Handles RGBA by compositing on white background.
        """
        image = Image.open(path)
        
        if image.mode == 'RGBA':
            # Composite on white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # Use alpha as mask
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        return np.array(image)
    
    def _load_mask(
        self,
        image_id: str,
        original_size: Tuple[int, int],
        label: Optional[int] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Load mask from .npy file or create zero mask.
        
        IMPORTANT: Authentic images (label=0) ALWAYS get zero masks,
        even if a mask file exists with the same name.
        
        In new structure (dataset_split/):
            - Images: authentic_12345.png, forged_12345.png
            - Masks: authentic_12345.npy (zero mask), forged_12345.npy (real mask)
            - image_id will be "authentic_12345" or "forged_12345"
        
        Args:
            image_id: Image ID (filename without extension)
                      In new structure: "authentic_12345" or "forged_12345"
                      In old structure: "12345"
            original_size: (H, W) of the original image
            label: Optional label (0=authentic, 1=forged). If provided and 0, always return zeros.
            
        Returns:
            mask: (H, W, num_mask_channels) numpy array
            valid_channels: List indicating which channels have valid data
        """
        H, W = original_size
        valid_channels = [0] * self.num_mask_channels
        
        # Check if this is an authentic image from the filename (new structure)
        # In new structure: image_id is "authentic_12345" or "forged_12345"
        # In old structure: image_id is "12345"
        is_authentic_from_filename = image_id.startswith('authentic_')
        
        # If explicitly labeled as authentic OR filename indicates authentic, return zero mask
        if label == 0 or is_authentic_from_filename:
            # Authentic images always get zero masks
            # Even if a mask file exists (in new structure, it's a saved zero mask),
            # we return zeros directly for consistency
            return np.zeros((H, W, self.num_mask_channels), dtype=np.uint8), valid_channels
        
        if self.mask_dir is None or self.is_test:
            # No masks available - return zeros
            return np.zeros((H, W, self.num_mask_channels), dtype=np.uint8), valid_channels
        
        # Determine mask filename
        # New structure: image_id is "authentic_12345" or "forged_12345", mask is "{image_id}.npy"
        # Old structure: image_id is "12345", mask is "{image_id}.npy"
        # So we can use image_id directly!
        mask_filename = f"{image_id}.npy"
        
        # Check if mask exists (handle both Path and CombinedMaskDir)
        if isinstance(self.mask_dir, CombinedMaskDir):
            mask_exists = False
            mask_path = None
            
            for mask_dir in self.mask_dir.mask_dirs:
                potential_path = mask_dir / mask_filename
                if potential_path.exists():
                    mask_path = potential_path
                    mask_exists = True
                    break
        else:
            mask_path = self.mask_dir / mask_filename
            mask_exists = mask_path.exists()
        
        if not mask_exists:
            # No mask found - treat as authentic (return zeros)
            # This should only happen in old structure or edge cases
            return np.zeros((H, W, self.num_mask_channels), dtype=np.uint8), valid_channels
        
        # Load mask
        mask = np.load(mask_path)  # Shape: (C, H, W) where C is 1, 2, or 3
        
        # Handle different mask shapes
        if mask.ndim == 2:
            mask = mask[np.newaxis, ...]  # (H, W) -> (1, H, W)
        
        # Convert from (C, H, W) to (H, W, C)
        mask = np.transpose(mask, (1, 2, 0))
        
        # Resize mask to match original image size if needed
        if mask.shape[:2] != original_size:
            mask = cv2.resize(
                mask, (W, H),
                interpolation=cv2.INTER_NEAREST
            )
            if mask.ndim == 2:
                mask = mask[:, :, np.newaxis]
        
        # Convert multi-channel mask to single channel by taking max across channels
        # This combines all regions from different channels into a single binary mask
        if mask.shape[2] > 1:
            # Take max across channels to get union of all regions
            mask = np.max(mask, axis=2, keepdims=True)  # (H, W, 1)
        
        # Ensure single channel output
        if mask.shape[2] != 1:
            mask = mask[:, :, :1] if mask.shape[2] > 1 else mask
        
        # Track valid channels (always 1 for single channel)
        valid_channels = [1] if self.num_mask_channels >= 1 else []
        
        # Pad to num_mask_channels if needed (should be 1)
        if mask.shape[2] < self.num_mask_channels:
            padding = np.zeros(
                (H, W, self.num_mask_channels - mask.shape[2]),
                dtype=mask.dtype
            )
            mask = np.concatenate([mask, padding], axis=2)
        elif mask.shape[2] > self.num_mask_channels:
            mask = mask[:, :, :self.num_mask_channels]
        
        return mask.astype(np.uint8), valid_channels


def get_train_transforms(image_size: Tuple[int, int] = (512, 512)) -> A.Compose:
    """
    Get training augmentations.
    
    Includes augmentations that preserve forgery characteristics
    while adding robustness to the model.
    """
    return A.Compose([
        # Resize
        A.Resize(height=image_size[0], width=image_size[1]),
        
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=30,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        
        # Color transforms
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ], p=0.5),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.1)),  # Normalized to [0, 1]
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MedianBlur(blur_limit=3),
        ], p=0.3),
        
        # JPEG compression (realistic for web images)
        A.ImageCompression(quality_range=(70, 95), p=0.3),
        
        # Normalize and convert to tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transforms(image_size: Tuple[int, int] = (512, 512)) -> A.Compose:
    """
    Get validation/test transforms (no augmentation).
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def create_data_splits(
    data_root: Path,
    train_ratio: float = 0.9,
    seed: int = 42,
    split_dir: Optional[Path] = None
) -> Tuple[List[Path], List[int], List[Path], List[int]]:
    """
    Create train/val splits with stratification.
    
    Uses existing split directory if available (dataset_split or split_data),
    otherwise creates a simple random split.
    
    Returns:
        train_paths, train_labels, val_paths, val_labels
    """
    import random
    random.seed(seed)
    
    data_root = Path(data_root)
    
    # Check for pre-computed splits
    if split_dir is None:
        # Try new structure first
        split_dir = data_root.parent / "dataset_split"
        if not split_dir.exists():
            # Fallback to old structure
            split_dir = data_root.parent / "split_data"
    
    if split_dir and split_dir.exists():
        return _load_clustered_split(split_dir, data_root)
    
    # Otherwise, simple stratified split
    return _create_simple_split(data_root, train_ratio, seed)


def _load_clustered_split(
    split_dir: Path,
    data_root: Path
) -> Tuple[List[Path], List[int], List[Path], List[int]]:
    """
    Load splits from new dataset_split structure or old split_data structure.
    
    New structure (dataset_split/):
        - train/authentic_*.png, forged_*.png
        - val/authentic_*.png, forged_*.png
        - train_masks/, val_masks/
    
    Old structure (split_data/):
        - train/cluster_*/*.png
        - val/cluster_*/*.png
    """
    # Check for new structure first
    csv_path = split_dir / "clustering_info.csv"
    if csv_path.exists():
        return _load_from_dataset_split(split_dir, csv_path)
    
    # Fallback to old structure
    return _load_from_old_structure(split_dir, data_root)


def _load_from_dataset_split(
    split_dir: Path,
    csv_path: Path
) -> Tuple[List[Path], List[int], List[Path], List[int]]:
    """Load from new dataset_split structure with CSV info"""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    
    # Get paths and labels from new structure
    train_paths = [split_dir / "train" / row['new_filename'] for _, row in train_df.iterrows()]
    train_labels = [1 if row['image_type'] == 'forged' else 0 for _, row in train_df.iterrows()]
    
    val_paths = [split_dir / "val" / row['new_filename'] for _, row in val_df.iterrows()]
    val_labels = [1 if row['image_type'] == 'forged' else 0 for _, row in val_df.iterrows()]
    
    # Verify files exist
    train_paths = [p for p in train_paths if p.exists()]
    val_paths = [p for p in val_paths if p.exists()]
    
    return train_paths, train_labels, val_paths, val_labels


def _load_from_old_structure(
    split_dir: Path,
    data_root: Path
) -> Tuple[List[Path], List[int], List[Path], List[int]]:
    """Load from old split_data structure"""
    train_dir = split_dir / "train"
    val_dir = split_dir / "val"
    
    masks_dir = data_root / "train_masks"
    supplemental_masks = data_root / "supplemental_masks"
    
    def collect_from_split(base_dir):
        paths = []
        labels = []
        
        for cluster_dir in base_dir.iterdir():
            if not cluster_dir.is_dir():
                continue
            
            for img_path in cluster_dir.glob("*.png"):
                image_id = img_path.stem
                
                # Check if mask exists (forged) or not (authentic)
                mask_exists = (
                    (masks_dir / f"{image_id}.npy").exists() or
                    (supplemental_masks / f"{image_id}.npy").exists()
                )
                
                paths.append(img_path)
                labels.append(1 if mask_exists else 0)
        
        return paths, labels
    
    train_paths, train_labels = collect_from_split(train_dir)
    val_paths, val_labels = collect_from_split(val_dir)
    
    return train_paths, train_labels, val_paths, val_labels


def _create_simple_split(
    data_root: Path,
    train_ratio: float,
    seed: int
) -> Tuple[List[Path], List[int], List[Path], List[int]]:
    """Create simple stratified random split"""
    from sklearn.model_selection import train_test_split
    
    # Collect all images
    forged_dir = data_root / "train_images" / "forged"
    authentic_dir = data_root / "train_images" / "authentic"
    supplemental_dir = data_root / "supplemental_images"
    
    all_paths = []
    all_labels = []
    
    # Forged images
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        for img in forged_dir.glob(ext):
            all_paths.append(img)
            all_labels.append(1)
    
    # Authentic images
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        for img in authentic_dir.glob(ext):
            all_paths.append(img)
            all_labels.append(0)
    
    # Supplemental (all forged)
    if supplemental_dir.exists():
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for img in supplemental_dir.glob(ext):
                all_paths.append(img)
                all_labels.append(1)
    
    # Split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels,
        train_size=train_ratio,
        stratify=all_labels,
        random_state=seed
    )
    
    return train_paths, train_labels, val_paths, val_labels


def get_combined_mask_dir(data_root: Path) -> Path:
    """
    Get or create a combined mask directory that includes both
    train_masks and supplemental_masks.
    
    For simplicity, we'll create a MaskDirectory class instead.
    """
    return data_root / "train_masks"


class CombinedMaskDir:
    """
    Virtual directory that looks up masks from multiple locations.
    """
    
    def __init__(self, *mask_dirs: Path):
        self.mask_dirs = [Path(d) for d in mask_dirs if Path(d).exists()]
    
    def __truediv__(self, filename: str) -> Path:
        """Support path / filename syntax"""
        for mask_dir in self.mask_dirs:
            path = mask_dir / filename
            if path.exists():
                return path
        # Return first dir path (will show as not existing)
        return self.mask_dirs[0] / filename if self.mask_dirs else Path(filename)


def create_dataloaders(
    data_root: Union[str, Path],
    batch_size: int = 4,
    image_size: Tuple[int, int] = (512, 512),
    num_workers: int = 4,
    train_ratio: float = 0.9,
    seed: int = 42,
    combine_mask_channels: bool = True,  # Always True - single channel masks
    dataset_split_dir: Optional[Union[str, Path]] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_root: Path to data directory
        batch_size: Batch size
        image_size: Target image size (H, W)
        num_workers: Number of data loading workers
        train_ratio: Train/val split ratio
        seed: Random seed
        
    Returns:
        train_loader, val_loader
    """
    data_root = Path(data_root)
    
    # Determine split directory - prioritize dataset_split structure
    if dataset_split_dir:
        split_dir = Path(dataset_split_dir)
    else:
        # Always try dataset_split first
        split_dir = data_root.parent / "dataset_split"
        if not split_dir.exists():
            # Fallback to old structure
            split_dir = data_root.parent / "split_data"
    
    # Create splits - prefer dataset_split structure
    if split_dir.exists() and (split_dir / "clustering_info.csv").exists():
        # Use new dataset_split structure
        train_paths, train_labels, val_paths, val_labels = create_data_splits(
            data_root, train_ratio, seed, split_dir=split_dir
        )
        
        # Use mask directories from dataset_split structure
        train_mask_dir = split_dir / "train_masks"
        val_mask_dir = split_dir / "val_masks"
        print(f"Using dataset_split structure: {split_dir}")
    else:
        # Use old structure or create new split
        train_paths, train_labels, val_paths, val_labels = create_data_splits(
            data_root, train_ratio, seed, split_dir=split_dir if split_dir.exists() else None
        )
        
        # Combined mask directory from original location
        train_mask_dir = CombinedMaskDir(
            data_root / "train_masks",
            data_root / "supplemental_masks"
        )
        val_mask_dir = train_mask_dir
        print(f"Using old structure or fallback")
    
    print(f"Training samples: {len(train_paths)} ({sum(train_labels)} forged, {len(train_labels) - sum(train_labels)} authentic)")
    print(f"Validation samples: {len(val_paths)} ({sum(val_labels)} forged, {len(val_labels) - sum(val_labels)} authentic)")
    print(f"Mask channels: single channel (max across all channels)")
    
    # Create datasets - always use single channel masks
    train_dataset = ForgeryDetectionDataset(
        image_paths=train_paths,
        mask_dir=train_mask_dir,
        labels=train_labels,
        transform=get_train_transforms(image_size),
        image_size=image_size,
        combine_mask_channels=True,  # Always True
        num_mask_channels=1  # Single channel
    )
    
    val_dataset = ForgeryDetectionDataset(
        image_paths=val_paths,
        mask_dir=val_mask_dir,
        labels=val_labels,
        transform=get_val_transforms(image_size),
        image_size=image_size,
        combine_mask_channels=True,  # Always True
        num_mask_channels=1  # Single channel
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_test_dataloader(
    test_dir: Union[str, Path],
    batch_size: int = 4,
    image_size: Tuple[int, int] = (512, 512),
    num_workers: int = 4
) -> DataLoader:
    """
    Create test dataloader for inference.
    """
    test_dir = Path(test_dir)
    
    # Collect test images
    test_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        test_paths.extend(list(test_dir.glob(ext)))
    
    print(f"Test samples: {len(test_paths)}")
    
    test_dataset = ForgeryDetectionDataset(
        image_paths=test_paths,
        mask_dir=None,
        labels=None,
        transform=get_val_transforms(image_size),
        image_size=image_size,
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader


if __name__ == "__main__":
    # Test dataset
    data_root = Path("recodai-luc-scientific-image-forgery-detection")
    
    train_loader, val_loader = create_dataloaders(
        data_root,
        batch_size=4,
        image_size=(512, 512)
    )
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"Image shape: {batch['image'].shape}")
    print(f"Mask shape: {batch['mask'].shape}")
    print(f"Labels: {batch['label']}")
    print(f"Valid channels: {batch['valid_mask_channels']}")
    print(f"Image IDs: {batch['image_id']}")

