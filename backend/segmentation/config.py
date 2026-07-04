import os
import torch

# Paths
MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "../weights/unet_face_celeb.pth")

# Model Architecture
NUM_CLASSES = 19  # 18 facial features + 1 background
INPUT_SIZE = (512, 512)
CROP_SIZE = (448, 448)

# Training Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Face Parsed Attributes (total 18 components, 1-indexed)
# 1: skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye, 6: eye_g, 7: l_ear, 8: r_ear, 9: ear_r, 10: nose, 11: mouth, 12: u_lip, 13: l_lip
# (Classes 14-18 are: 14: neck, 15: neck_l, 16: cloth, 17: hair, 18: hat)
# We define FACE_CLASSES to form the binary foreground mask.
FACE_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
