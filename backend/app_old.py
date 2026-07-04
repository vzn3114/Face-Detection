import os
import io
import base64
import numpy as np
import cv2
import torch
from flask import Flask, request, render_template, jsonify

from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
from segmentation.predict import UNetPredictor
from quality.face_quality import analyze_face_quality

app = Flask(__name__)

# Cache loaded models
models_cache = {}
unet_predictor = None

def get_unet_predictor():
    global unet_predictor
    if unet_predictor is None:
        unet_predictor = UNetPredictor()
    return unet_predictor

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print("remove prefix '{}'".format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model_instance(network_name):
    if network_name == "mobile0.25":
        cfg = cfg_mnet
        trained_model = './weights/mobilenet0.25_Final.pth'
    else:
        cfg = cfg_re50
        trained_model = './weights/Resnet50_Final.pth'
        
    net = RetinaFace(cfg=cfg, phase='test')
    
    # Check if the weight file exists
    if not os.path.exists(trained_model):
        raise FileNotFoundError(f"Weight file '{trained_model}' not found. Please download it first.")
        
    # Load to CPU
    pretrained_dict = torch.load(trained_model, map_location=lambda storage, loc: storage)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(net, pretrained_dict)
    net.load_state_dict(pretrained_dict, strict=False)
    net.eval()
    return net, cfg

def rle_encode(mask):
    """
    Encode binary mask to RLE format.
    mask: 2D numpy array (0 or 255/1).
    Returns a dict with RLE data.
    """
    flat = (mask > 0).astype(np.uint8).flatten()
    # Find transitions
    changes = np.diff(flat)
    idxs = np.where(changes != 0)[0] + 1
    # Add start and end indices
    idxs = np.concatenate(([0], idxs, [len(flat)]))
    runs = np.diff(idxs)
    
    return {
        'size': list(mask.shape),
        'counts': runs.tolist(),
        'start_val': int(flat[0])
    }

def segment_face(img_raw, box, landmarks):
    """
    Generate a precise face segmentation mask using GrabCut and Skin Color models,
    guided by the bounding box and facial landmarks.
    """
    h, w = img_raw.shape[:2]
    
    # Clip bounding box to image boundaries
    x1, y1, x2, y2 = map(int, box[:4])
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    rect_w = x2 - x1
    rect_h = y2 - y1
    
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Fallback to ellipse for very small regions to save compute
    if rect_w * rect_h < 400 or rect_w <= 4 or rect_h <= 4:
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        axes = (int(rect_w / 2), int(rect_h / 2))
        cv2.ellipse(binary_mask, center, axes, 0, 0, 360, 255, -1)
    else:
        try:
            # Extract face crop directly from the image
            face_crop = img_raw[y1:y2, x1:x2]
            # Run U-Net prediction
            predictor = get_unet_predictor()
            crop_mask = predictor.predict(face_crop)
            # Place the crop mask back to full-image binary mask
            binary_mask[y1:y2, x1:x2] = crop_mask
        except Exception as e:
            # Fallback to simple ellipse if U-Net prediction fails
            print("U-Net segmentation failed, falling back to ellipse:", e)
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            axes = (int(rect_w / 2), int(rect_h / 2))
            cv2.ellipse(binary_mask, center, axes, 0, 0, 360, 255, -1)
            
    # Generate cropped face with alpha channel
    # Include padding around bounding box for better visualization
    pad = min(15, int(max(rect_w, rect_h) * 0.1))
    px1 = max(0, x1 - pad)
    py1 = max(0, y1 - pad)
    px2 = min(w, x2 + pad)
    py2 = min(h, y2 + pad)
    
    crop_w = px2 - px1
    crop_h = py2 - py1
    
    if crop_w > 0 and crop_h > 0:
        face_crop = img_raw[py1:py2, px1:px2]
        mask_crop = binary_mask[py1:py2, px1:px2]
        
        # Smooth mask crop edges with Gaussian blur
        mask_crop_soft = cv2.GaussianBlur(mask_crop, (5, 5), 0)
        
        b_ch, g_ch, r_ch = cv2.split(face_crop)
        rgba_crop = cv2.merge([b_ch, g_ch, r_ch, mask_crop_soft])
        
        _, png_buffer = cv2.imencode('.png', rgba_crop)
        png_base64 = base64.b64encode(png_buffer).decode('utf-8')
        face_png_alpha = f"data:image/png;base64,{png_base64}"
    else:
        face_png_alpha = ""
        
    return binary_mask, face_png_alpha

def detect_faces(img_raw, network_name="mobile0.25", confidence_threshold=0.5, nms_threshold=0.4, draw_mask=False):
    if network_name not in models_cache:
        models_cache[network_name] = load_model_instance(network_name)
    net, cfg = models_cache[network_name]
    
    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    
    # We are running on CPU
    device = torch.device("cpu")
    net = net.to(device)
    img = img.to(device)
    scale = scale.to(device)
    
    with torch.no_grad():
        loc, conf, landms = net(img)  # forward pass
        
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1
    landms = landms.cpu().numpy()
    
    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]
    
    # keep top-K before NMS
    top_k = 5000
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]
    
    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]
    
    # keep top-K after NMS
    keep_top_k = 750
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]
    
    dets = np.concatenate((dets, landms), axis=1)
    
    # Draw results
    annotated_img = img_raw.copy()
    face_count = 0
    faces_data = []
    
    # We will use a purple-neon mask overlay
    overlay_color = [246, 92, 139] # BGR for purple (139, 92, 246)
    
    for i, b in enumerate(dets):
        if b[4] < confidence_threshold:
            continue
        face_count += 1
        confidence = float(b[4])
        box_coords = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
        landmarks_coords = [float(b[5+j]) for j in range(10)]
        
        # Segment face
        binary_mask, face_png_alpha = segment_face(img_raw, box_coords, landmarks_coords)
        
        # Run Face Quality Analysis
        q_results = analyze_face_quality(box_coords, confidence, landmarks_coords, binary_mask)
        
        # Encode RLE
        mask_rle = rle_encode(binary_mask)
        
        # Draw mask if requested
        if draw_mask:
            idx = (binary_mask > 0)
            if np.any(idx):
                mask_color_img = np.zeros_like(annotated_img)
                mask_color_img[idx] = overlay_color
                cv2.addWeighted(annotated_img, 0.7, mask_color_img, 0.3, 0, dst=annotated_img)
                
        # Draw bounding box and landmarks
        box_ints = list(map(int, box_coords))
        cv2.rectangle(annotated_img, (box_ints[0], box_ints[1]), (box_ints[2], box_ints[3]), (0, 255, 0), 2)
        
        # ID and confidence label
        text = f"ID:{face_count} {confidence:.2f}"
        cx = box_ints[0]
        cy = box_ints[1] - 5 if box_ints[1] - 5 > 15 else box_ints[1] + 15
        
        label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.4, 1)
        cv2.rectangle(annotated_img, (box_ints[0], cy - label_size[1] - 2), (box_ints[0] + label_size[0], cy + base_line - 2), (0, 255, 0), cv2.FILLED)
        cv2.putText(annotated_img, text, (box_ints[0], cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Landmarks (5 points)
        cv2.circle(annotated_img, (int(landmarks_coords[0]), int(landmarks_coords[1])), 3, (0, 0, 255), -1)
        cv2.circle(annotated_img, (int(landmarks_coords[2]), int(landmarks_coords[3])), 3, (0, 255, 255), -1)
        cv2.circle(annotated_img, (int(landmarks_coords[4]), int(landmarks_coords[5])), 3, (255, 0, 255), -1)
        cv2.circle(annotated_img, (int(landmarks_coords[6]), int(landmarks_coords[7])), 3, (0, 255, 0), -1)
        cv2.circle(annotated_img, (int(landmarks_coords[8]), int(landmarks_coords[9])), 3, (255, 0, 0), -1)
        
        faces_data.append({
            'id': face_count,
            'box': box_coords,
            'confidence': confidence,
            'landmarks': landmarks_coords,
            'mask_rle': mask_rle,
            'face_png_alpha': face_png_alpha,
            'visibility': q_results['visibility'],
            'quality_score': q_results['quality_score'],
            'pose': q_results['pose'],
            'status': q_results['status'],
            'rating': q_results['rating']
        })
        
    return annotated_img, face_count, faces_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'Chưa chọn file ảnh'}), 400
        
    file = request.files['image']
    network = request.form.get('network', 'mobile0.25')
    threshold = float(request.form.get('threshold', 0.5))
    draw_mask = request.form.get('draw_mask', 'false').lower() == 'true'
    
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    img_raw = cv2.imdecode(data, cv2.IMREAD_COLOR)
    
    if img_raw is None:
        return jsonify({'error': 'File ảnh không hợp lệ'}), 400
        
    try:
        # Run face detection and segmentation
        annotated_img, face_count, faces_data = detect_faces(
            img_raw, 
            network_name=network, 
            confidence_threshold=threshold, 
            draw_mask=draw_mask
        )
        
        # Convert output image back to base64 to send to frontend
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'image': f"data:image/jpeg;base64,{img_base64}",
            'face_count': face_count,
            'faces': faces_data
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server... Please open http://127.0.0.1:5000 in your browser.")
    app.run(host='0.0.0.0', port=5000, debug=True)
