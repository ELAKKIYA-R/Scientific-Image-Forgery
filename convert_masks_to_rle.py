#!/usr/bin/env python
# coding: utf-8

"""
Convert RFDETRSegPreview model predictions to RLE format.
Handles both forged images (with masks) and authentic images (empty masks).
"""

import numpy as np
from metric import rle_encode


def prediction_to_rle(prediction):
    """
    Convert RFDETRSegPreview prediction masks to RLE format.
    
    Args:
        prediction: Detections object from RFDETRSegPreview.predict()
                   with attributes:
                   - mask: numpy array of shape (num_detections, height, width) 
                          with boolean values (True for mask, False for background)
    
    Returns:
        str: RLE-encoded string if masks are present, or 'authentic' if no masks
    """
    masks = prediction.mask
    
    # Handle authentic images (no detections)
    if masks.size == 0 or len(masks) == 0:
        return 'authentic'
    
    # Convert boolean masks to integer arrays (1 for mask, 0 for background)
    # masks is shape (num_detections, height, width) with boolean values
    mask_list = []
    for i in range(len(masks)):
        # Convert boolean to int: True -> 1, False -> 0
        mask_int = masks[i].astype(np.int32)
        mask_list.append(mask_int)
    
    # Encode to RLE format
    rle_string = rle_encode(mask_list, fg_val=1)
    
    return rle_string


# Example usage:
if __name__ == '__main__':
    from rfdetr import RFDETRSegPreview
    from PIL import Image
    
    # Initialize model
    model = RFDETRSegPreview(pretrain_weights='/kaggle/input/rfdetr-seg/pytorch/default/1/checkpoint0017.pth')
    
    # Load and process image
    img_path = '/kaggle/input/recodai-luc-scientific-image-forgery-detection/train_images/forged/10070.png'
    img = Image.open(img_path)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Get prediction
    prediction = model.predict(img)
    
    # Convert to RLE
    rle_result = prediction_to_rle(prediction)
    print(f"RLE result: {rle_result}")

