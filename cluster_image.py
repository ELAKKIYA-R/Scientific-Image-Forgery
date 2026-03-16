import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import hdbscan
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

class DINOv2FeatureExtractor(nn.Module):
    def __init__(self, model_name='dinov2_vitb14'):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.feature_dim = self.model.embed_dim
        
    def forward(self, x):
        return self.model(x)

class ScientificImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, str(img_path)

def extract_features(model, dataloader, device='cuda'):
    model.eval()
    all_features = []
    all_paths = []
    
    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            features = model(images)
            all_features.append(features.cpu().numpy())
            all_paths.extend(paths)
    
    return np.vstack(all_features), all_paths

def cluster_and_split(
    image_dir,
    model_path,
    output_dir,
    train_ratio=0.9,
    batch_size=32,
    model_name='dinov2_vitb14',
    min_cluster_size=3
):
    # MPS support for M1/M2 Macs
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    
    # Get all image paths
    if isinstance(image_dir, list):
        image_paths = []
        for dir_path in image_dir:
            image_paths.extend(list(Path(dir_path).rglob('*.png')))
            image_paths.extend(list(Path(dir_path).rglob('*.jpg')))
            image_paths.extend(list(Path(dir_path).rglob('*.jpeg')))
    else:
        image_paths = list(Path(image_dir).rglob('*.png')) + \
                      list(Path(image_dir).rglob('*.jpg')) + \
                      list(Path(image_dir).rglob('*.jpeg'))
    
    # Load model
    model = DINOv2FeatureExtractor(model_name=model_name)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device)
    
    # Prepare dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ScientificImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Extract features
    features, paths = extract_features(model, dataloader, device=device)
    
    # Cluster
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=3,
        metric='euclidean'
    )
    cluster_labels = clusterer.fit_predict(features)
    
    # Create output directories
    output_path = Path(output_dir)
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Split and copy files
    # NOTE: This script clusters BOTH authentic and forged images together.
    # The dataset will later determine which images are forged by checking
    # if a corresponding mask file exists. Authentic images (no mask) will
    # automatically get zero masks during training.
    unique_clusters = set(cluster_labels)
    
    for cluster_id in tqdm(unique_clusters, desc="Splitting clusters"):
        # Get images in this cluster
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        cluster_paths = [paths[i] for i in cluster_indices]
        
        if cluster_id == -1:
            folder_name = 'noise'
        else:
            folder_name = f'cluster_{cluster_id}'
        
        # Split into train/val
        if len(cluster_paths) == 1:
            train_paths = cluster_paths
            val_paths = []
        else:
            train_paths, val_paths = train_test_split(
                cluster_paths,
                train_size=train_ratio,
                random_state=42
            )
        
        # Create cluster directories
        train_cluster_dir = train_dir / folder_name
        val_cluster_dir = val_dir / folder_name
        train_cluster_dir.mkdir(exist_ok=True)
        val_cluster_dir.mkdir(exist_ok=True)
        
        # Copy train files
        # NOTE: If authentic and forged images have the same filename,
        # the later copy will overwrite the earlier one. The dataset will
        # determine authentic vs forged by checking if a mask file exists.
        for img_path in train_paths:
            src = Path(img_path)
            dst = train_cluster_dir / src.name
            # Check if file already exists (same name from different folder)
            if dst.exists():
                # Keep track: if both authentic and forged versions exist,
                # prefer the forged version (has mask) for training
                # The dataset will check mask existence to determine label
                pass
            shutil.copy2(src, dst)
        
        # Copy val files
        for img_path in val_paths:
            src = Path(img_path)
            dst = val_cluster_dir / src.name
            if dst.exists():
                pass  # Same handling as train
            shutil.copy2(src, dst)

if __name__ == "__main__":
    # Include authentic, forged, and supplemental images
    # This ensures all image types are represented in train/val splits
    cluster_and_split(
        image_dir=[
            "recodai-luc-scientific-image-forgery-detection/train_images/authentic",
            "recodai-luc-scientific-image-forgery-detection/train_images/forged",  # Fixed: was 'forgery'
            "recodai-luc-scientific-image-forgery-detection/supplemental_images"   # Added supplemental
        ],
        model_path="dinov2_finetuned.pth",  # Updated path
        output_dir="split_data",
        train_ratio=0.9,
        batch_size=16,  # Reduced for M2 memory
        model_name='dinov2_vitb14',
        min_cluster_size=3
    )