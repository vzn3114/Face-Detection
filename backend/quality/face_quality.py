import numpy as np

def check_face_size(box, img_shape):
    """
    Evaluate face size relative to typical recognition standards.
    """
    x1, y1, x2, y2 = box[:4]
    w = x2 - x1
    h = y2 - y1
    
    face_dim = max(w, h)
    if face_dim < 36:
        return "Too Small"
    elif face_dim > 256:
        return "Large"
    return "Normal"

def check_visibility(binary_mask, box):
    """
    Calculate the ratio of the segmented face mask within the bounding box.
    Returns visibility percentage (0-100).
    """
    h, w = binary_mask.shape[:2]
    x1, y1, x2, y2 = map(int, box[:4])
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    bbox_area = (x2 - x1) * (y2 - y1)
    if bbox_area <= 0:
        return 0
        
    # Count segmented face pixels inside the bounding box
    mask_crop = binary_mask[y1:y2, x1:x2]
    mask_area = np.sum(mask_crop > 0)
    
    visibility = float(mask_area) / float(bbox_area)
    return min(100, int(visibility * 100))

def check_occlusion(visibility_pct):
    """
    Check if the face is occluded based on mask visibility percentage.
    """
    if visibility_pct < 65:
        return True
    return False

def check_head_pose(landmarks):
    """
    Geometry-based head pose estimation using 5 facial landmarks:
    [left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y, mouth_left_x, mouth_left_y, mouth_right_x, mouth_right_y]
    """
    if not landmarks or len(landmarks) < 10:
        return "Normal"
        
    lx, ly = landmarks[0], landmarks[1]
    rx, ry = landmarks[2], landmarks[3]
    nx, ny = landmarks[4], landmarks[5]
    mlx, mly = landmarks[6], landmarks[7]
    mrx, mry = landmarks[8], landmarks[9]
    
    # 1. Pitch estimation (Head Up / Head Down)
    eye_y = (ly + ry) / 2.0
    mouth_y = (mly + mry) / 2.0
    dist_eye_mouth = mouth_y - eye_y
    dist_eye_nose = ny - eye_y
    
    if dist_eye_mouth <= 0:
        return "Normal"
        
    nose_y_ratio = dist_eye_nose / dist_eye_mouth
    if nose_y_ratio > 0.82:
        return "Head Down"
    elif nose_y_ratio < 0.28:
        return "Head Up"
        
    # 2. Yaw estimation (Head Left / Head Right)
    dist_eyes_x = rx - lx
    if dist_eyes_x <= 0:
        return "Normal"
        
    dist_left_eye_nose_x = nx - lx
    nose_x_ratio = dist_left_eye_nose_x / dist_eyes_x
    if nose_x_ratio < 0.33:
        return "Head Left"
    elif nose_x_ratio > 0.67:
        return "Head Right"
        
    return "Normal"

def analyze_face_quality(box, confidence, landmarks, binary_mask):
    """
    Runs all face checks, computes a composite score (0-100), and flags status.
    """
    # 1. Individual metrics checks
    size_status = check_face_size(box, binary_mask.shape)
    visibility_pct = check_visibility(binary_mask, box)
    is_occluded = check_occlusion(visibility_pct)
    pose = check_head_pose(landmarks)
    
    # 2. Compute components scores
    det_score = confidence * 100
    vis_score = visibility_pct
    
    # Size component score
    if size_status == "Too Small":
        size_score = 40
    elif size_status == "Large":
        size_score = 95
    else:
        size_score = 100
        
    # Pose component score
    if pose == "Normal":
        pose_score = 100
    else:
        pose_score = 75
        
    # 3. Calculate composite Quality Score
    quality_score = int(0.3 * det_score + 0.4 * vis_score + 0.2 * size_score + 0.1 * pose_score)
    quality_score = max(0, min(100, quality_score))
    
    # 4. Resolve Quality Status rating & descriptive alerts
    status_msg = "Normal"
    if size_status == "Too Small":
        status_msg = "Face Too Small"
        quality_score = min(35, quality_score) # Cap score for unusable size
    elif confidence < 0.65:
        status_msg = "Low Confidence"
        quality_score = min(35, quality_score)
    elif is_occluded:
        status_msg = "Face Occluded"
        quality_score = min(50, quality_score) # Cap score for occlusion
    elif pose != "Normal":
        status_msg = pose
        
    # Rating levels
    if quality_score >= 85:
        rating = "Excellent"
    elif quality_score >= 70:
        rating = "Good"
    elif quality_score >= 55:
        rating = "Acceptable"
    elif quality_score >= 40:
        rating = "Poor"
    else:
        rating = "Unusable"
        
    return {
        'visibility': visibility_pct,
        'quality_score': quality_score,
        'pose': pose,
        'status': status_msg,
        'rating': rating
    }
