import os
import cv2
import numpy as np
import onnxruntime as ort

from segmentation.config import INPUT_SIZE, FACE_CLASSES
from segmentation.postprocess import logits_to_binary_mask

class UNetFaceSegmenter:
    def __init__(self, weights_path=None):
        if weights_path is None:
            weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../weights/unet_face_celeb.onnx"))
            
        print(f"Loading U-Net Face Segmenter ONNX model from {weights_path}...")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(weights_path, sess_options=opts, providers=['CPUExecutionProvider'])
        print("U-Net Face Segmenter ONNX model loaded successfully.")

        # Input image normalization constants (ImageNet standards)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def predict_batch(self, face_crops_bgr):
        """
        Segment a list of cropped face images using U-Net ONNX in a single batch.
        
        Args:
            face_crops_bgr (list of np.ndarray): list of BGR image patches
            
        Returns:
            list of np.ndarray: list of Binary masks matching the shapes of respective input crops
        """
        if not face_crops_bgr:
            return []
            
        valid_indices = []
        batch_list = []
        original_sizes = []
        
        for idx, crop in enumerate(face_crops_bgr):
            h_orig, w_orig = crop.shape[:2]
            original_sizes.append((h_orig, w_orig))
            if h_orig == 0 or w_orig == 0:
                continue
                
            valid_indices.append(idx)
            # Convert BGR (OpenCV) to RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Resize to U-Net input size (512x512)
            resized_img = cv2.resize(crop_rgb, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
            
            # Scale to [0, 1] and normalize using numpy
            img_normalized = (resized_img.astype(np.float32) / 255.0 - self.mean) / self.std
            
            # Transpose to (3, H, W)
            img_tensor = img_normalized.transpose(2, 0, 1)
            batch_list.append(img_tensor)
            
        # Initialize output masks with empty arrays matching original sizes
        out_masks = [np.zeros((sz[0], sz[1]), dtype=np.uint8) for sz in original_sizes]
        
        if not batch_list:
            return out_masks
            
        # Stack into shape (N, 3, 512, 512)
        batch_tensor = np.stack(batch_list, axis=0)
        
        # Run ONNX inference in batch
        input_name = self.session.get_inputs()[0].name
        logits = self.session.run(None, {input_name: batch_tensor})[0]  # shape (N, 19, 512, 512)
        
        # Postprocess each mask
        for i, val_idx in enumerate(valid_indices):
            single_logits = logits[i]  # shape (19, 512, 512)
            mask_512 = logits_to_binary_mask(single_logits)
            h_orig, w_orig = original_sizes[val_idx]
            mask_orig = cv2.resize(mask_512, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            out_masks[val_idx] = mask_orig
            
        return out_masks

    def predict(self, face_crop_bgr):
        """
        Segment a cropped face image using U-Net ONNX.
        
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
        
        # Scale to [0, 1] and normalize using numpy
        img_normalized = (resized_img.astype(np.float32) / 255.0 - self.mean) / self.std
        
        # Transpose to (3, H, W) and add batch dimension (1, 3, H, W)
        img_tensor = img_normalized.transpose(2, 0, 1)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        
        # Run ONNX inference
        input_name = self.session.get_inputs()[0].name
        logits = self.session.run(None, {input_name: img_tensor})[0]  # shape (1, 19, 512, 512)
        
        # Convert logits to binary mask (512x512)
        binary_mask_512 = logits_to_binary_mask(logits)
        
        # Resize back to original crop size using nearest neighbor (to preserve binary classes)
        binary_mask_orig = cv2.resize(binary_mask_512, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        
        return binary_mask_orig
