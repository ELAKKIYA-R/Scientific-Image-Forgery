"""
Utility functions for Copy-Move Forgery Detection
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import cv2


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving"""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6
) -> float:
    """
    Compute Intersection over Union.
    
    Args:
        pred: Predicted binary mask (B, C, H, W) or (B, H, W)
        target: Ground truth mask (B, C, H, W) or (B, H, W)
        smooth: Smoothing factor
        
    Returns:
        IoU score
    """
    pred = pred.float().flatten()
    target = target.float().flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def compute_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6
) -> float:
    """
    Compute Dice coefficient.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth mask
        smooth: Smoothing factor
        
    Returns:
        Dice score
    """
    pred = pred.float().flatten()
    target = target.float().flatten()
    
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def compute_metrics(
    mask_pred: torch.Tensor,
    mask_target: torch.Tensor,
    cls_pred: torch.Tensor,
    cls_target: torch.Tensor
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        mask_pred: Predicted masks (B, C, H, W)
        mask_target: Ground truth masks (B, C, H, W)
        cls_pred: Predicted classes (B,)
        cls_target: Ground truth classes (B,)
        
    Returns:
        Dictionary of metrics
    """
    # Segmentation metrics
    iou = compute_iou(mask_pred, mask_target)
    dice = compute_dice(mask_pred, mask_target)
    
    # Classification metrics
    correct = (cls_pred == cls_target).sum().item()
    total = cls_target.numel()
    accuracy = correct / total
    
    # Precision, recall, F1 for classification
    # Assuming class 1 is "forged"
    tp = ((cls_pred == 1) & (cls_target == 1)).sum().item()
    fp = ((cls_pred == 1) & (cls_target == 0)).sum().item()
    fn = ((cls_pred == 0) & (cls_target == 1)).sum().item()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        'iou': iou,
        'dice': dice,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    losses: Dict[str, float],
    metrics: Dict[str, float],
    filepath: Union[str, Path]
):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'losses': losses,
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = torch.device('cpu')
) -> Tuple[nn.Module, int, Dict]:
    """Load training checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    return model, epoch, metrics


def visualize_prediction(
    image: torch.Tensor,
    mask_pred: torch.Tensor,
    mask_gt: Optional[torch.Tensor] = None,
    correlation_map: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None
):
    """
    Visualize model predictions.
    
    Args:
        image: Input image (C, H, W) - normalized
        mask_pred: Predicted mask (C, H, W)
        mask_gt: Ground truth mask (C, H, W) - optional
        correlation_map: Correlation heatmap (1, H, W) - optional
        save_path: Path to save figure
    """
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = image.permute(1, 2, 0).numpy()
    image = np.clip(image, 0, 1)
    
    # Convert masks
    mask_pred = mask_pred.numpy()
    if mask_gt is not None:
        mask_gt = mask_gt.numpy()
    
    # Setup figure
    n_cols = 3 if mask_gt is not None else 2
    if correlation_map is not None:
        n_cols += 1
    
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Predicted mask (combine channels)
    pred_combined = mask_pred.max(axis=0)
    axes[1].imshow(pred_combined, cmap='hot')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')
    
    col_idx = 2
    
    # Ground truth
    if mask_gt is not None:
        gt_combined = mask_gt.max(axis=0)
        axes[col_idx].imshow(gt_combined, cmap='hot')
        axes[col_idx].set_title('Ground Truth')
        axes[col_idx].axis('off')
        col_idx += 1
    
    # Correlation map
    if correlation_map is not None:
        corr = correlation_map[0].numpy()
        axes[col_idx].imshow(corr, cmap='viridis')
        axes[col_idx].set_title('Correlation Heatmap')
        axes[col_idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_batch(
    images: torch.Tensor,
    mask_preds: torch.Tensor,
    mask_gts: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    cls_preds: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    max_samples: int = 8
):
    """
    Visualize a batch of predictions.
    """
    batch_size = min(images.shape[0], max_samples)
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = images * std + mean
    
    n_cols = 3
    fig, axes = plt.subplots(batch_size, n_cols, figsize=(4 * n_cols, 4 * batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Image
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[i, 0].imshow(img)
        
        title = 'Input'
        if labels is not None:
            gt_label = 'Forged' if labels[i] == 1 else 'Authentic'
            title += f'\nGT: {gt_label}'
        if cls_preds is not None:
            pred_label = 'Forged' if cls_preds[i] == 1 else 'Authentic'
            title += f'\nPred: {pred_label}'
        axes[i, 0].set_title(title)
        axes[i, 0].axis('off')
        
        # Predicted mask
        pred = mask_preds[i].max(dim=0)[0].numpy()
        axes[i, 1].imshow(pred, cmap='hot')
        axes[i, 1].set_title('Predicted Mask')
        axes[i, 1].axis('off')
        
        # Ground truth mask
        if mask_gts is not None:
            gt = mask_gts[i].max(dim=0)[0].numpy()
            axes[i, 2].imshow(gt, cmap='hot')
            axes[i, 2].set_title('Ground Truth')
            axes[i, 2].axis('off')
        else:
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """
    Overlay mask on image with transparency.
    
    Args:
        image: RGB image (H, W, 3)
        mask: Binary mask (H, W)
        alpha: Transparency
        color: Overlay color (R, G, B)
        
    Returns:
        Image with overlay
    """
    overlay = image.copy()
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    
    # Blend
    overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
    
    return overlay


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count model parameters.
    
    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in megabytes"""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

