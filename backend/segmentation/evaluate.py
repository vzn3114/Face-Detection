import os
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torchvision.transforms as transforms

from segmentation.config import MODEL_WEIGHTS_PATH, NUM_CLASSES, INPUT_SIZE, DEVICE
from segmentation.model import UNet
from segmentation.dataset import FaceMask

# Part colors for 20 classes (0: background + 18 classes, plus buffer)
PART_COLORS = [
    [0, 0, 0], [255, 0, 0], [255, 85, 0], [255, 170, 0],
    [255, 0, 85], [255, 0, 170], [0, 255, 0], [85, 255, 0],
    [170, 255, 0], [0, 255, 85], [0, 255, 170], [0, 0, 255],
    [85, 0, 255], [170, 0, 255], [0, 85, 255], [0, 170, 255],
    [255, 255, 0], [255, 255, 85], [255, 255, 170], [255, 0, 255]
]

def vis_parsing_maps(im, parsing_anno, save_im=True, save_path='parsing_map.jpg'):
    """
    Generate and save colored semantic parsing overlay.
    """
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    
    # Resize parsing anno to match image dimensions
    vis_parsing_anno = cv2.resize(parsing_anno, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3), dtype=np.uint8) + 255

    num_of_class = np.max(vis_parsing_anno)
    for pi in range(1, min(num_of_class + 1, len(PART_COLORS))):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = PART_COLORS[pi]

    # Overlay with opacity (40% original, 60% color parsing map)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    if save_im:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
    return vis_im

def compute_iou(pred, label, num_classes):
    """
    Compute Intersection-over-Union (IoU) for all classes.
    """
    iou_list = []
    for class_idx in range(num_classes):
        pred_mask = (pred == class_idx)
        label_mask = (label == class_idx)
        
        intersection = np.logical_and(pred_mask, label_mask).sum()
        union = np.logical_or(pred_mask, label_mask).sum()
        
        if union == 0:
            iou_list.append(float('nan'))  # Class not present in either ground truth or prediction
        else:
            iou_list.append(intersection / union)
            
    return np.array(iou_list)

def evaluate(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load Model
    print("Loading U-Net model...")
    model = UNet(n_channels=3, n_classes=NUM_CLASSES)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print("Model weights loaded successfully.")
    else:
        print(f"Error: Weights file not found at {args.weights}")
        return

    model.to(device)
    model.eval()
    
    # Load Dataset (test/val mode)
    print("Loading test dataset...")
    full_dataset = FaceMask(mode='test')
    
    if args.limit is not None:
        subset_indices = list(range(min(args.limit, len(full_dataset))))
        dataset = Subset(full_dataset, subset_indices)
    else:
        dataset = full_dataset
        
    print(f"Evaluating on {len(dataset)} samples...")
    
    # Run evaluation
    accuracies = []
    ious = []
    
    results_dir = './res/unet_test_res'
    os.makedirs(results_dir, exist_ok=True)
    
    # Transforms to reload clean raw image for visualizer
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            # Retrieve from raw full dataset to get filename
            img_idx = idx if args.limit is None else args.limit
            if idx >= len(dataset):
                break
                
            # Dataset item returns normalized tensor and label
            img_tensor, label_tensor = dataset[idx]
            
            # Forward pass
            img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)
            out = model(img_tensor)
            
            # Predict parsing map
            parsing = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()  # (H, W)
            target = label_tensor[0]  # (H, W)
            
            # Compute pixel accuracy
            correct = (parsing == target).sum()
            total = target.size
            accuracies.append(correct / total)
            
            # Compute IoUs
            iou = compute_iou(parsing, target, NUM_CLASSES)
            ious.append(iou)
            
            # Visual overlay save for first 15 images
            if idx < 15:
                # Load raw image
                impth = dataset.dataset.imgs[idx] if hasattr(dataset, 'dataset') else dataset.imgs[idx]
                image_dir = dataset.dataset.image_dir if hasattr(dataset, 'dataset') else dataset.image_dir
                raw_img = Image.open(os.path.join(image_dir, impth)).resize(INPUT_SIZE, Image.BILINEAR)
                
                save_path = os.path.join(results_dir, impth)
                vis_parsing_maps(raw_img, parsing, save_im=True, save_path=save_path)
                
    # Calculate statistics
    mean_acc = np.mean(accuracies)
    mean_ious = np.nanmean(np.array(ious), axis=0)
    miou = np.nanmean(mean_ious)
    
    print(f"\n--- Evaluation Results ---")
    print(f"Mean Pixel Accuracy: {mean_acc * 100:.2f}%")
    print(f"Mean IoU (mIoU) across 19 classes: {miou * 100:.2f}%")
    print(f"Validation visual overlays saved to {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate U-Net Face Segmentation")
    parser.add_argument("--weights", type=str, default=MODEL_WEIGHTS_PATH, help="Path to weights file")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device (cuda or cpu)")
    parser.add_argument("--limit", type=int, default=15, help="Number of images to evaluate")
    args = parser.parse_args()
    
    evaluate(args)
