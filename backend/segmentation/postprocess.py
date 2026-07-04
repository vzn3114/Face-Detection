import numpy as np
import torch
from segmentation.config import FACE_CLASSES

def logits_to_binary_mask(logits):
    """
    Convert model output logits to a binary mask where face components are 255 and background/others are 0.
    
    Args:
        logits (torch.Tensor or np.ndarray): Raw logits from U-Net model of shape (19, H, W) or (1, 19, H, W)
        
    Returns:
        np.ndarray: Binary mask of shape (H, W) with values 0 or 255 (np.uint8)
    """
    if isinstance(logits, np.ndarray):
        if len(logits.shape) == 4:  # (1, 19, H, W)
            logits = logits[0]
        preds = np.argmax(logits, axis=0)  # (H, W)
        binary_mask = np.zeros_like(preds, dtype=np.uint8)
        for class_idx in FACE_CLASSES:
            binary_mask[preds == class_idx] = 255
    else:
        # Assuming PyTorch Tensor
        if len(logits.shape) == 4:  # (1, 19, H, W)
            logits = logits.squeeze(0)
        
        with torch.no_grad():
            preds = torch.argmax(logits, dim=0).cpu().numpy()  # (H, W)
            
        binary_mask = np.zeros_like(preds, dtype=np.uint8)
        for class_idx in FACE_CLASSES:
            binary_mask[preds == class_idx] = 255
            
    return binary_mask
