"""
Loss functions for Copy-Move Forgery Detection

Includes:
- Dice Loss for segmentation
- BCE Loss with optional weighting
- Focal Loss for class imbalance
- Combined multi-task loss
- Correlation consistency loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Handles multi-channel masks with optional per-channel weighting.
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        reduction: str = 'mean',
        ignore_empty_channels: bool = True
    ):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: 'mean', 'sum', or 'none'
            ignore_empty_channels: If True, ignore channels with no GT
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_empty_channels = ignore_empty_channels
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_channels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred: Predictions (B, C, H, W) - raw logits
            target: Ground truth (B, C, H, W)
            valid_channels: Optional (B, C) tensor indicating valid channels
            
        Returns:
            Scalar loss value
        """
        # Apply sigmoid to logits
        pred = torch.sigmoid(pred)
        
        B, C, H, W = pred.shape
        
        # Flatten spatial dimensions
        pred_flat = pred.view(B, C, -1)
        target_flat = target.view(B, C, -1)
        
        # Compute dice per channel
        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Handle empty masks (all zeros): if both pred and target are empty, dice should be 1.0
        # This is already handled by the formula above when union = 0 (both empty)
        
        # Apply channel masking if provided
        if valid_channels is not None:
            # valid_channels shape: (B, C)
            dice = dice * valid_channels
            
            if self.ignore_empty_channels:
                # Average only over valid channels
                num_valid = valid_channels.sum(dim=1).clamp(min=1)
                dice = dice.sum(dim=1) / num_valid
            else:
                dice = dice.mean(dim=1)
        else:
            dice = dice.mean(dim=1)
        
        # Dice loss = 1 - dice coefficient
        # For empty masks (all zeros), if prediction is also all zeros:
        #   intersection = 0, union = 0, dice = smooth/smooth = 1.0, loss = 0.0 ✓
        # If prediction has non-zero values for empty target:
        #   intersection = 0, union > 0, dice < 1.0, loss > 0.0 ✓
        loss = 1.0 - dice
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BCEDiceLoss(nn.Module):
    """
    Combined BCE and Dice loss for segmentation.
    """
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        pos_weight: Optional[float] = None
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        self.dice = DiceLoss(smooth=smooth)
        
        # BCE with optional positive class weighting
        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight])
            )
        else:
            self.bce = nn.BCEWithLogitsLoss()
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_channels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute combined BCE + Dice loss"""
        dice_loss = self.dice(pred, target, valid_channels)
        
        # For BCE, we need to handle valid channels differently
        if valid_channels is not None:
            # Mask out invalid channels
            mask = valid_channels.unsqueeze(-1).unsqueeze(-1)
            mask = mask.expand_as(pred)
            
            # Only compute BCE where mask is valid
            valid_pred = pred * mask
            valid_target = target * mask
            
            bce_loss = self.bce(valid_pred, valid_target)
        else:
            bce_loss = self.bce(pred, target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in classification.
    
    FL(p) = -α * (1 - p)^γ * log(p)
    
    Supports per-class alpha values for better calibration.
    """
    
    def __init__(
        self,
        alpha: Optional[Union[float, Tuple[float, float]]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        num_classes: int = 2
    ):
        """
        Args:
            alpha: If float, single alpha for both classes.
                   If tuple (alpha_0, alpha_1), per-class alpha values.
                   If None, no alpha weighting (alpha=1 for all classes).
                   Typical: (0.25, 0.75) means class 0 gets 0.25 weight, class 1 gets 0.75 weight.
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
            num_classes: Number of classes (default 2 for binary classification)
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes
        
        # Handle alpha: convert to per-class tensor if needed
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (int, float)):
            # Single alpha value: use for class 1, (1-alpha) for class 0
            # This maintains backward compatibility
            self.alpha = torch.tensor([1.0 - alpha, alpha], dtype=torch.float32)
        elif isinstance(alpha, (tuple, list)) and len(alpha) == num_classes:
            # Per-class alpha values
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            raise ValueError(f"alpha must be float, tuple of {num_classes} floats, or None")
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            pred: Predictions (B, num_classes) - raw logits
            target: Ground truth (B,) - class indices (0=authentic, 1=forged)
            
        Returns:
            Scalar loss value
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        p = F.softmax(pred, dim=1)
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply per-class alpha weighting
        if self.alpha is not None:
            # Move alpha to same device as predictions
            if self.alpha.device != pred.device:
                self.alpha = self.alpha.to(pred.device)
            # Get alpha for each sample based on its class
            alpha_t = self.alpha[target]  # (B,)
            focal_weight = alpha_t * focal_weight
        
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CorrelationConsistencyLoss(nn.Module):
    """
    Loss to encourage correlation map to match forgery regions.
    
    The correlation heatmap should highlight regions that are
    duplicated (copy-move forgery).
    """
    
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
        
    def forward(
        self,
        correlation_heatmap: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute correlation consistency loss.
        
        Args:
            correlation_heatmap: Predicted correlation (B, 1, H, W)
            mask: Ground truth mask (B, C, H, W)
            
        Returns:
            Scalar loss value
        """
        # Resize correlation to mask size if needed
        if correlation_heatmap.shape[2:] != mask.shape[2:]:
            correlation_heatmap = F.interpolate(
                correlation_heatmap,
                size=mask.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Create binary target from mask (any channel)
        # Forged regions should have high correlation
        target = mask.max(dim=1, keepdim=True)[0]  # (B, 1, H, W)
        
        # MSE loss between correlation and target
        loss = F.mse_loss(correlation_heatmap, target.float())
        
        return loss


class CMFDLoss(nn.Module):
    """
    Combined multi-task loss for Copy-Move Forgery Detection.
    
    Total loss = w1 * seg_loss + w2 * cls_loss + w3 * corr_loss
    
    Where:
    - seg_loss: BCE + Dice for mask segmentation
    - cls_loss: Focal loss for authentic/forged classification
    - corr_loss: Correlation consistency loss (optional)
    """
    
    def __init__(
        self,
        seg_weight: float = 1.0,
        cls_weight: float = 0.5,
        corr_weight: float = 0.1,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        dice_smooth: float = 1.0,
        focal_alpha: Optional[Union[float, Tuple[float, float]]] = None,
        focal_gamma: float = 2.0,
        use_correlation_loss: bool = True
    ):
        super().__init__()
        
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.corr_weight = corr_weight
        self.use_correlation_loss = use_correlation_loss
        
        # Segmentation loss
        self.seg_loss = BCEDiceLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            smooth=dice_smooth
        )
        
        # Classification loss
        # Default: if focal_alpha is None, use balanced weighting (0.5, 0.5)
        # For class imbalance, use (alpha_authentic, alpha_forged)
        # Example: (0.25, 0.75) gives more weight to forged class (minority)
        if focal_alpha is None:
            focal_alpha = (0.5, 0.5)  # Balanced by default
        
        self.cls_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            num_classes=2
        )
        
        # Correlation loss
        if use_correlation_loss:
            self.corr_loss = CorrelationConsistencyLoss()
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            outputs: Model outputs with keys:
                - 'mask_logits': (B, C, H, W)
                - 'class_logits': (B, 2)
                - 'correlation_heatmap': (B, 1, H', W') - optional
            targets: Ground truth with keys:
                - 'mask': (B, C, H, W)
                - 'label': (B,)
                - 'valid_mask_channels': (B, C)
                
        Returns:
            total_loss: Combined scalar loss
            loss_dict: Dictionary of individual losses for logging
        """
        loss_dict = {}
        
        # Segmentation loss
        seg_loss = self.seg_loss(
            outputs['mask_logits'],
            targets['mask'],
            targets.get('valid_mask_channels')
        )
        loss_dict['seg_loss'] = seg_loss
        
        # Classification loss
        cls_loss = self.cls_loss(
            outputs['class_logits'],
            targets['label']
        )
        loss_dict['cls_loss'] = cls_loss
        
        # Correlation loss
        corr_loss = torch.tensor(0.0, device=seg_loss.device)
        if self.use_correlation_loss and 'correlation_heatmap' in outputs:
            corr_loss = self.corr_loss(
                outputs['correlation_heatmap'],
                targets['mask']
            )
            loss_dict['corr_loss'] = corr_loss
        
        # Total loss
        total_loss = (
            self.seg_weight * seg_loss +
            self.cls_weight * cls_loss +
            self.corr_weight * corr_loss
        )
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict


class IoULoss(nn.Module):
    """
    IoU Loss for segmentation.
    Alternative to Dice Loss.
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute IoU loss"""
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice Loss.
    
    Useful for handling class imbalance in segmentation.
    alpha > beta penalizes false positives more
    alpha < beta penalizes false negatives more
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute Tversky loss"""
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # True positives, false positives, false negatives
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1 - tversky


def get_loss_function(config) -> CMFDLoss:
    """
    Factory function to create loss function from config.
    """
    return CMFDLoss(
        seg_weight=config.training.segmentation_loss_weight,
        cls_weight=config.training.classification_loss_weight,
        corr_weight=config.training.correlation_loss_weight,
        dice_smooth=config.training.dice_smooth,
        focal_alpha=getattr(config.training, 'focal_alpha', None),
        use_correlation_loss=config.model.use_self_correlation
    )

