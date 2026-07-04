import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from segmentation.config import MODEL_WEIGHTS_PATH, NUM_CLASSES, INPUT_SIZE, DEVICE
from segmentation.model import UNet
from segmentation.postprocess import logits_to_binary_mask

class UNetPredictor:
    def __init__(self, weights_path=MODEL_WEIGHTS_PATH, device=DEVICE):
        self.device = torch.device(device)
        self.model = UNet(n_channels=3, n_classes=NUM_CLASSES)
        
        if os.path.exists(weights_path):
            print(f"Loading U-Net segmentation weights from {weights_path}...")
            # Load weights to CPU/configured device
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("Segmentation weights loaded successfully.")
        else:
            print(f"WARNING: U-Net segmentation weights not found at {weights_path}.")
            print("Model will run with random initialization. Please run the training pipeline first.")
            
        self.model.to(self.device)
        self.model.eval()

        # Input image normalization (ImageNet standards)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def predict(self, face_crop_bgr):
        """
        Segment a cropped face image.
        
        Args:
            face_crop_bgr (np.ndarray): BGR image patch cropped from the original image (OpenCV format)
            
        Returns:
            np.ndarray: Binary mask matching the shape of face_crop_bgr (values 0 or 255)
        """
        h_orig, w_orig = face_crop_bgr.shape[:2]
        if h_orig == 0 or w_orig == 0:
            return np.zeros((h_orig, w_orig), dtype=np.uint8)
            
        # Convert BGR (OpenCV) to RGB
        face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        
        # Resize to U-Net input size (512x512)
        resized_img = cv2.resize(face_crop_rgb, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # Convert to float and scale to [0, 1]
        img_tensor = torch.from_numpy(resized_img.transpose(2, 0, 1)).float() / 255.0
        
        # Normalize
        img_tensor = self.normalize(img_tensor)
        
        # Add batch dimension and send to device
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(img_tensor)  # shape (1, 19, 512, 512)
            
        # Convert logits to binary mask (512x512)
        binary_mask_512 = logits_to_binary_mask(logits)
        
        # Resize back to original crop size using nearest neighbor (to preserve binary classes)
        binary_mask_orig = cv2.resize(binary_mask_512, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        
        return binary_mask_orig
