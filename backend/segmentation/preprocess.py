import os
import cv2
import numpy as np
from PIL import Image
import argparse
from concurrent.futures import ProcessPoolExecutor
from segmentation.config import MASK_ANNO_DIR, MASK_DIR

# Attributes list (1-indexed mapping, total 18)
ATTRIBUTES = [
    'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
    'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat'
]

def preprocess_single_mask(j):
    """
    Combine separate annotation masks for image index j into a single 19-class mask.
    Saves the final mask as j.png in the mask output directory.
    """
    out_mask_path = os.path.join(MASK_DIR, f"{j}.png")
    
    # Initialize 512x512 mask with zeros (background)
    mask = np.zeros((512, 512), dtype=np.uint8)
    folder_idx = j // 2000
    
    has_any = False
    for l, att in enumerate(ATTRIBUTES, 1):
        file_name = f"{str(j).rjust(5, '0')}_{att}.png"
        path = os.path.join(MASK_ANNO_DIR, str(folder_idx), file_name)
        
        if os.path.exists(path):
            has_any = True
            # Load mask image in grayscale/palette
            sep_mask = np.array(Image.open(path).convert('P'))
            # Pixels with value 225 indicate the attribute class presence
            mask[sep_mask == 225] = l
            
    # Save the combined mask
    cv2.imwrite(out_mask_path, mask)
    return j, has_any

def main():
    parser = argparse.ArgumentParser(description="Preprocess CelebAMask-HQ separate annotations into a combined multi-class mask.")
    parser.add_argument("--limit", type=int, default=None, help="Limit preprocessing to first N images.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel worker processes.")
    args = parser.parse_args()
    
    if not os.path.exists(MASK_DIR):
        os.makedirs(MASK_DIR, exist_ok=True)
        print(f"Created mask directory at {MASK_DIR}")
        
    num_images = 30000
    if args.limit is not None:
        num_images = min(num_images, args.limit)
        
    print(f"Starting preprocessing of {num_images} masks using {args.workers} workers...")
    
    indices = list(range(num_images))
    
    count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(preprocess_single_mask, indices)
        for j, has_any in results:
            count += 1
            if count % 2000 == 0 or count == num_images:
                print(f"Preprocessed {count}/{num_images} masks...")

    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
