import os
import json
import time
import cv2

def save_evidence(img_raw, face_crop, binary_mask_crop, face_metadata, output_dir="evidence"):
    """
    Saves surveillance evidence: full frame, face crop, binary mask, and metadata JSON.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    face_id = face_metadata.get('id', 0)
    
    # Define file names
    frame_name = f"frame_{timestamp}_face{face_id}.jpg"
    crop_name = f"face_{timestamp}_face{face_id}.png"
    mask_name = f"mask_{timestamp}_face{face_id}.png"
    meta_name = f"meta_{timestamp}_face{face_id}.json"
    
    # Save files
    cv2.imwrite(os.path.join(output_dir, frame_name), img_raw)
    cv2.imwrite(os.path.join(output_dir, crop_name), face_crop)
    cv2.imwrite(os.path.join(output_dir, mask_name), binary_mask_crop)
    
    with open(os.path.join(output_dir, meta_name), 'w', encoding='utf-8') as f:
        json.dump(face_metadata, f, indent=4, ensure_ascii=False)
        
    return {
        'frame_path': os.path.join(output_dir, frame_name),
        'crop_path': os.path.join(output_dir, crop_name),
        'mask_path': os.path.join(output_dir, mask_name),
        'meta_path': os.path.join(output_dir, meta_name)
    }
