import os
import sys
import io
import base64
import time
import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Add backend directory to sys.path so inner imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from models.face_detector import RetinaFaceDetector
from models.face_segmenter import UNetFaceSegmenter
from quality.face_quality import analyze_face_quality

app = FastAPI(title="AI Crowd Face Surveillance API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache detectors and segmenter
detectors = {}
segmenter = None

def get_detector(network_name: str):
    if network_name not in detectors:
        detectors[network_name] = RetinaFaceDetector(network_name=network_name)
    return detectors[network_name]

def get_segmenter():
    global segmenter
    if segmenter is None:
        segmenter = UNetFaceSegmenter()
    return segmenter

def rle_encode(mask):
    flat = (mask > 0).astype(np.uint8).flatten()
    changes = np.diff(flat)
    idxs = np.where(changes != 0)[0] + 1
    idxs = np.concatenate(([0], idxs, [len(flat)]))
    runs = np.diff(idxs)
    return {
        'size': list(mask.shape),
        'counts': runs.tolist(),
        'start_val': int(flat[0])
    }

def segment_face(img_raw, box, landmarks):
    h, w = img_raw.shape[:2]
    x1, y1, x2, y2 = map(int, box[:4])
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    rect_w = x2 - x1
    rect_h = y2 - y1
    
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    
    if rect_w * rect_h < 400 or rect_w <= 4 or rect_h <= 4:
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        axes = (int(rect_w / 2), int(rect_h / 2))
        cv2.ellipse(binary_mask, center, axes, 0, 0, 360, 255, -1)
    else:
        try:
            face_crop = img_raw[y1:y2, x1:x2]
            seg = get_segmenter()
            crop_mask = seg.predict(face_crop)
            binary_mask[y1:y2, x1:x2] = crop_mask
        except Exception as e:
            print("U-Net segmentation failed, falling back to ellipse:", e)
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            axes = (int(rect_w / 2), int(rect_h / 2))
            cv2.ellipse(binary_mask, center, axes, 0, 0, 360, 255, -1)
            
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
        mask_crop_soft = cv2.GaussianBlur(mask_crop, (5, 5), 0)
        
        b_ch, g_ch, r_ch = cv2.split(face_crop)
        rgba_crop = cv2.merge([b_ch, g_ch, r_ch, mask_crop_soft])
        
        _, png_buffer = cv2.imencode('.png', rgba_crop)
        png_base64 = base64.b64encode(png_buffer).decode('utf-8')
        face_png_alpha = f"data:image/png;base64,{png_base64}"
    else:
        face_png_alpha = ""
        
    return binary_mask, face_png_alpha

@app.get("/", response_class=HTMLResponse)
def get_index():
    index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend/index.html"))
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Giao diện Frontend không tìm thấy.")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    network: str = Form("mobile0.25"),
    threshold: float = Form(0.5),
    draw_mask: bool = Form(False),
    upscale: bool = Form(False)
):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_raw is None:
            raise HTTPException(status_code=400, detail="File ảnh không hợp lệ")
            
        t_start = time.time()
        
        detector = get_detector(network)
        faces_list = detector.detect(img_raw, confidence_threshold=threshold, upscale=upscale)
        
        h, w = img_raw.shape[:2]
        
        # 1. Collect crops for batched U-Net segmentation
        unet_crops = []
        unet_indices = []
        
        for idx, face in enumerate(faces_list):
            box_coords = face['box']
            x1, y1, x2, y2 = map(int, box_coords[:4])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            rect_w = x2 - x1
            rect_h = y2 - y1
            
            if rect_w * rect_h < 400 or rect_w <= 4 or rect_h <= 4:
                continue
            
            face_crop = img_raw[y1:y2, x1:x2]
            unet_crops.append(face_crop)
            unet_indices.append(idx)
            
        # 2. Run batched U-Net segmentation
        unet_masks = []
        if unet_crops:
            try:
                unet_masks = get_segmenter().predict_batch(unet_crops)
            except Exception as e:
                print("U-Net batch segmentation failed, fallback to ellipses:", e)
                unet_masks = []
                
        # 3. Assemble results and calculate face metrics
        annotated_img = img_raw.copy()
        overlay_color = [246, 92, 139] # BGR purple (139, 92, 246)
        
        faces_data = []
        face_count = 0
        unet_map = {face_idx: mask_idx for mask_idx, face_idx in enumerate(unet_indices)}
        
        for idx, face in enumerate(faces_list):
            face_count += 1
            box_coords = face['box']
            confidence = face['confidence']
            landmarks_coords = face['landmarks']
            
            x1, y1, x2, y2 = map(int, box_coords[:4])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            rect_w = x2 - x1
            rect_h = y2 - y1
            
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Map U-Net mask output or fallback to ellipse
            if idx in unet_map and unet_map[idx] < len(unet_masks):
                crop_mask = unet_masks[unet_map[idx]]
                binary_mask[y1:y2, x1:x2] = crop_mask
            else:
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                axes = (int(rect_w / 2), int(rect_h / 2))
                cv2.ellipse(binary_mask, center, axes, 0, 0, 360, 255, -1)
                
            # Generate alpha-blended transparent face crop
            pad = min(15, int(max(rect_w, rect_h) * 0.1))
            px1 = max(0, x1 - pad)
            py1 = max(0, y1 - pad)
            px2 = min(w, x2 + pad)
            py2 = min(h, y2 + pad)
            
            crop_w = px2 - px1
            crop_h = py2 - py1
            
            face_png_alpha = ""
            if crop_w > 0 and crop_h > 0:
                face_crop_region = img_raw[py1:py2, px1:px2]
                mask_crop = binary_mask[py1:py2, px1:px2]
                mask_crop_soft = cv2.GaussianBlur(mask_crop, (5, 5), 0)
                
                b_ch, g_ch, r_ch = cv2.split(face_crop_region)
                rgba_crop = cv2.merge([b_ch, g_ch, r_ch, mask_crop_soft])
                
                _, png_buffer = cv2.imencode('.png', rgba_crop)
                png_base64 = base64.b64encode(png_buffer).decode('utf-8')
                face_png_alpha = f"data:image/png;base64,{png_base64}"
                
            # Quality assessment
            q_results = analyze_face_quality(box_coords, confidence, landmarks_coords, binary_mask)
            mask_rle = rle_encode(binary_mask)
            
            # Draw mask overlay
            if draw_mask:
                idx_pts = (binary_mask > 0)
                if np.any(idx_pts):
                    mask_color_img = np.zeros_like(annotated_img)
                    mask_color_img[idx_pts] = overlay_color
                    cv2.addWeighted(annotated_img, 0.7, mask_color_img, 0.3, 0, dst=annotated_img)
                    
            # Draw bounding box and landmarks
            box_ints = list(map(int, box_coords))
            cv2.rectangle(annotated_img, (box_ints[0], box_ints[1]), (box_ints[2], box_ints[3]), (0, 255, 0), 2)
            
            # Label box
            text = f"ID:{face_count} {confidence:.2f}"
            cx = box_ints[0]
            cy = box_ints[1] - 5 if box_ints[1] - 5 > 15 else box_ints[1] + 15
            label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.4, 1)
            cv2.rectangle(annotated_img, (box_ints[0], cy - label_size[1] - 2), (box_ints[0] + label_size[0], cy + base_line - 2), (0, 255, 0), cv2.FILLED)
            cv2.putText(annotated_img, text, (box_ints[0], cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            
            # Draw 5 landmarks
            for j in range(5):
                cv2.circle(annotated_img, (int(landmarks_coords[2*j]), int(landmarks_coords[2*j+1])), 3, (0, 0, 255), -1)
                
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
            
        t_latency = int((time.time() - t_start) * 1000)
        
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'image': f"data:image/jpeg;base64,{img_base64}",
            'face_count': face_count,
            'faces': faces_data,
            'latency_ms': t_latency
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files under /static
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend"))
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
