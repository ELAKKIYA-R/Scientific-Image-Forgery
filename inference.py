"""
Inference script for Copy-Move Forgery Detection

Generates submission file for Kaggle competition.
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from config import Config, get_config
from dataset import create_test_dataloader, get_val_transforms, ForgeryDetectionDataset
from models import CMFDNet
from utils import load_checkpoint


def load_model(
    checkpoint_path: str,
    config: Config,
    device: torch.device
) -> CMFDNet:
    """Load trained model from checkpoint"""
    model = CMFDNet(
        encoder_name=config.model.encoder_name,
        encoder_pretrained=False,  # Don't need pretrained for inference
        decoder_channels=config.model.decoder_channels,
        num_mask_classes=config.model.num_mask_classes,
        num_classification_classes=config.model.num_classification_classes,
        use_self_correlation=config.model.use_self_correlation,
        correlation_feature_scale=config.model.correlation_feature_scale,
        correlation_temperature=config.model.correlation_temperature,
        dropout=0.0,  # No dropout during inference
        use_gradient_checkpointing=False
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    if 'metrics' in checkpoint:
        print(f"  Checkpoint metrics: {checkpoint['metrics']}")
    
    return model


@torch.no_grad()
def predict_single_image(
    model: CMFDNet,
    image_path: Path,
    device: torch.device,
    image_size: Tuple[int, int] = (512, 512),
    threshold: float = 0.5
) -> Dict:
    """
    Run inference on a single image.
    
    Returns:
        Dictionary with:
            - 'class': 'authentic' or 'forged'
            - 'class_prob': probability of being forged
            - 'mask': binary mask (H, W, 1) at original resolution (single channel)
            - 'mask_prob': probability mask (H, W, 1)
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (W, H)
    
    # Apply transforms
    transform = get_val_transforms(image_size)
    transformed = transform(image=np.array(image))
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Forward pass
    outputs = model(image_tensor)
    
    # Classification
    class_probs = F.softmax(outputs['class_logits'], dim=1)
    class_pred = class_probs.argmax(dim=1).item()
    forged_prob = class_probs[0, 1].item()
    
    # Mask
    mask_probs = torch.sigmoid(outputs['mask_logits'])
    mask_pred = (mask_probs > threshold).float()
    
    # Resize mask to original resolution
    mask_probs = F.interpolate(
        mask_probs,
        size=(original_size[1], original_size[0]),  # (H, W)
        mode='bilinear',
        align_corners=False
    )
    mask_pred = F.interpolate(
        mask_pred,
        size=(original_size[1], original_size[0]),
        mode='nearest'
    )
    
    # Handle single channel mask output
    # Model outputs (B, C, H, W), after removing batch: (C, H, W) = (1, H, W)
    mask_pred_np = mask_pred[0].cpu().numpy()  # (1, H, W) or (C, H, W)
    mask_prob_np = mask_probs[0].cpu().numpy()  # (1, H, W) or (C, H, W)
    
    # Convert from (C, H, W) to (H, W, C) format for easier saving/visualization
    # Note: Ground truth masks from dataset are in (C, H, W) format
    # If comparing directly, convert one format to match the other
    if mask_pred_np.ndim == 2:
        # (H, W) -> (H, W, 1)
        mask_pred_np = mask_pred_np[:, :, np.newaxis]
    elif mask_pred_np.shape[0] == 1:
        # (1, H, W) -> (H, W, 1)
        mask_pred_np = mask_pred_np.transpose(1, 2, 0)
    elif mask_pred_np.ndim == 3 and mask_pred_np.shape[2] <= 3:
        # Already in (H, W, C) format or needs transpose
        if mask_pred_np.shape[0] <= 3 and mask_pred_np.shape[0] < mask_pred_np.shape[2]:
            # Likely (C, H, W), convert to (H, W, C)
            mask_pred_np = mask_pred_np.transpose(1, 2, 0)
    
    if mask_prob_np.ndim == 2:
        mask_prob_np = mask_prob_np[:, :, np.newaxis]
    elif mask_prob_np.shape[0] == 1:
        mask_prob_np = mask_prob_np.transpose(1, 2, 0)
    elif mask_prob_np.ndim == 3 and mask_prob_np.shape[2] <= 3:
        if mask_prob_np.shape[0] <= 3 and mask_prob_np.shape[0] < mask_prob_np.shape[2]:
            mask_prob_np = mask_prob_np.transpose(1, 2, 0)
    
    return {
        'class': 'forged' if class_pred == 1 else 'authentic',
        'class_prob': forged_prob,
        'mask': mask_pred_np,  # (H, W, 1) - Note: different from dataset format (C, H, W)
        'mask_prob': mask_prob_np  # (H, W, 1) - Note: different from dataset format (C, H, W)
    }


@torch.no_grad()
def predict_batch(
    model: CMFDNet,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[List[str], List[str], List[float]]:
    """
    Run inference on a batch of images.
    
    Returns:
        image_ids: List of image IDs
        predictions: List of predictions ('authentic' or 'forged')
        probabilities: List of forged probabilities
    """
    model.eval()
    
    all_image_ids = []
    all_predictions = []
    all_probabilities = []
    
    for batch in tqdm(dataloader, desc="Running inference"):
        images = batch['image'].to(device)
        image_ids = batch['image_id']
        
        # Forward pass
        outputs = model(images)
        
        # Classification
        class_probs = F.softmax(outputs['class_logits'], dim=1)
        class_preds = class_probs.argmax(dim=1)
        forged_probs = class_probs[:, 1]
        
        # Also check mask to refine prediction
        # If significant mask area, likely forged
        mask_probs = torch.sigmoid(outputs['mask_logits'])
        # Handle single channel mask (B, 1, H, W) or multi-channel (B, C, H, W)
        mask_area = (mask_probs > threshold).float().mean(dim=(1, 2, 3))
        
        # Combine classification and mask evidence
        # High mask area suggests forgery even if classifier is uncertain
        refined_preds = ((class_preds == 1) | (mask_area > 0.01)).long()
        
        for img_id, pred, prob in zip(
            image_ids,
            refined_preds.cpu().numpy(),
            forged_probs.cpu().numpy()
        ):
            all_image_ids.append(img_id)
            all_predictions.append('forged' if pred == 1 else 'authentic')
            all_probabilities.append(float(prob))
    
    return all_image_ids, all_predictions, all_probabilities


def generate_submission(
    model: CMFDNet,
    test_dir: Path,
    output_path: Path,
    config: Config,
    device: torch.device,
    threshold: float = 0.5
):
    """
    Generate submission CSV file.
    
    Expected format:
        case_id,annotation
        45,authentic
        ...
    """
    # Create test dataloader
    test_loader = create_test_dataloader(
        test_dir=test_dir,
        batch_size=config.training.batch_size,
        image_size=config.data.image_size,
        num_workers=config.training.num_workers
    )
    
    # Run inference
    image_ids, predictions, probabilities = predict_batch(
        model, test_loader, device, threshold
    )
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'case_id': image_ids,
        'annotation': predictions
    })
    
    # Sort by case_id (numeric)
    submission['case_id_int'] = submission['case_id'].astype(int)
    submission = submission.sort_values('case_id_int')
    submission = submission.drop('case_id_int', axis=1)
    
    # Save
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Total samples: {len(submission)}")
    print(f"Predictions: {submission['annotation'].value_counts().to_dict()}")
    
    # Also save probabilities for analysis
    prob_df = pd.DataFrame({
        'case_id': image_ids,
        'annotation': predictions,
        'forged_probability': probabilities
    })
    prob_path = output_path.parent / f"{output_path.stem}_with_probs.csv"
    prob_df.to_csv(prob_path, index=False)
    print(f"Probabilities saved to {prob_path}")


def save_mask_predictions(
    model: CMFDNet,
    test_dir: Path,
    output_dir: Path,
    config: Config,
    device: torch.device,
    threshold: float = 0.5
):
    """
    Save predicted masks as .npy files.
    
    Useful for detailed analysis or ensemble methods.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all test images
    test_images = list(Path(test_dir).glob('*.png'))
    
    for img_path in tqdm(test_images, desc="Saving masks"):
        result = predict_single_image(
            model, img_path, device,
            config.data.image_size, threshold
        )
        
        # Save mask
        mask_path = output_dir / f"{img_path.stem}.npy"
        np.save(mask_path, result['mask_prob'])
        
        # Save visualization (optional)
        if False:  # Set to True to save visualizations
            from utils import visualize_prediction
            viz_path = output_dir / f"{img_path.stem}_viz.png"
            # Load image for visualization
            image = Image.open(img_path).convert('RGB')
            # ... visualization code


def ensemble_predict(
    model_paths: List[str],
    test_dir: Path,
    output_path: Path,
    config: Config,
    device: torch.device
):
    """
    Ensemble prediction from multiple models.
    
    Uses average probability for final prediction.
    """
    # Load all models
    models = []
    for path in model_paths:
        model = load_model(path, config, device)
        models.append(model)
    
    print(f"Loaded {len(models)} models for ensemble")
    
    # Create test dataloader
    test_loader = create_test_dataloader(
        test_dir=test_dir,
        batch_size=config.training.batch_size,
        image_size=config.data.image_size,
        num_workers=config.training.num_workers
    )
    
    # Collect predictions from all models
    all_image_ids = None
    all_probs = []
    
    for model in models:
        model.eval()
        probs = []
        image_ids = []
        
        for batch in tqdm(test_loader, desc=f"Model inference"):
            images = batch['image'].to(device)
            
            with torch.no_grad():
                outputs = model(images)
                class_probs = F.softmax(outputs['class_logits'], dim=1)
                forged_probs = class_probs[:, 1]
            
            probs.extend(forged_probs.cpu().numpy().tolist())
            image_ids.extend(batch['image_id'])
        
        all_probs.append(probs)
        if all_image_ids is None:
            all_image_ids = image_ids
    
    # Average probabilities
    avg_probs = np.mean(all_probs, axis=0)
    predictions = ['forged' if p > 0.5 else 'authentic' for p in avg_probs]
    
    # Create submission
    submission = pd.DataFrame({
        'case_id': all_image_ids,
        'annotation': predictions
    })
    
    submission['case_id_int'] = submission['case_id'].astype(int)
    submission = submission.sort_values('case_id_int')
    submission = submission.drop('case_id_int', axis=1)
    
    submission.to_csv(output_path, index=False)
    print(f"Ensemble submission saved to {output_path}")


def main():
    """Entry point for inference"""
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--test_dir', type=str, 
        default='recodai-luc-scientific-image-forgery-detection/test_images',
        help='Path to test images directory'
    )
    parser.add_argument(
        '--output', type=str, default='submission.csv',
        help='Output submission file path'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Classification threshold'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        choices=['mps', 'cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--save_masks', action='store_true',
        help='Save predicted masks as .npy files'
    )
    parser.add_argument(
        '--masks_dir', type=str, default='predicted_masks',
        help='Directory to save predicted masks'
    )
    parser.add_argument(
        '--ensemble', type=str, nargs='+', default=None,
        help='List of checkpoint paths for ensemble'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = get_config()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    test_dir = Path(args.test_dir)
    output_path = Path(args.output)
    
    if args.ensemble:
        # Ensemble prediction
        ensemble_predict(
            args.ensemble, test_dir, output_path, config, device
        )
    else:
        # Single model prediction
        model = load_model(args.checkpoint, config, device)
        
        # Generate submission
        generate_submission(
            model, test_dir, output_path, config, device, args.threshold
        )
        
        # Optionally save masks
        if args.save_masks:
            save_mask_predictions(
                model, test_dir, Path(args.masks_dir),
                config, device, args.threshold
            )


if __name__ == "__main__":
    main()

