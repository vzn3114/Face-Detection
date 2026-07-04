import os
import cv2
import numpy as np
import torch
import onnxruntime as ort

from data.config import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms

class RetinaFaceDetector:
    def __init__(self, network_name="mobile0.25", weights_dir=None):
        self.network_name = network_name
        if weights_dir is None:
            weights_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../weights"))
            
        if network_name == "mobile0.25":
            self.cfg = cfg_mnet
            self.weights_path = os.path.join(weights_dir, "mobilenet0.25_Final.onnx")
        else:
            self.cfg = cfg_re50
            self.weights_path = os.path.join(weights_dir, "Resnet50_Final.onnx")
            
        print(f"Loading RetinaFace ONNX model from {self.weights_path}...")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(self.weights_path, sess_options=opts, providers=['CPUExecutionProvider'])
        print("RetinaFace ONNX model loaded successfully.")

    def detect(self, img_raw, confidence_threshold=0.5, nms_threshold=0.4, upscale=False):
        im_height, im_width, _ = img_raw.shape
        # Crowd detection optimization: upscale image if it's small and upscale=True to detect tiny/distant faces
        target_width = 1280
        scale_factor = 1.0
        if upscale and im_width < target_width:
            scale_factor = target_width / im_width
            new_width = int(im_width * scale_factor)
            new_height = int(im_height * scale_factor)
            img_processed = cv2.resize(img_raw, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:
            img_processed = img_raw
            
        proc_height, proc_width, _ = img_processed.shape
        
        # Preprocessing
        img = np.float32(img_processed)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0) # shape (1, 3, H, W)
        
        # Run ONNX inference
        input_name = self.session.get_inputs()[0].name
        loc, conf, landms = self.session.run(None, {input_name: img})
        
        # Convert outputs to torch tensors for decoding
        loc_t = torch.from_numpy(loc).squeeze(0)
        conf_t = torch.from_numpy(conf).squeeze(0)
        landms_t = torch.from_numpy(landms).squeeze(0)
        
        # Decoders (reusing existing pytorch functions)
        priorbox = PriorBox(self.cfg, image_size=(proc_height, proc_width))
        priors = priorbox.forward()
        prior_data = priors.data
        
        # Decode boxes
        boxes = decode(loc_t, prior_data, self.cfg['variance'])
        scale = torch.Tensor([proc_width, proc_height, proc_width, proc_height])
        boxes = boxes * scale
        boxes = boxes / scale_factor  # Scale back to original resolution
        boxes = boxes.cpu().numpy()
        
        # Decode scores
        scores = conf_t.cpu().numpy()[:, 1]
        
        # Decode landmarks
        landms = decode_landm(landms_t, prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([proc_width, proc_height, proc_width, proc_height,
                               proc_width, proc_height, proc_width, proc_height,
                               proc_width, proc_height])
        landms = landms * scale1
        landms = landms / scale_factor  # Scale back to original resolution
        landms = landms.cpu().numpy()
        
        # Filter low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        
        # Keep top-K before NMS
        top_k = 5000
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        
        # Do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]
        
        # Keep top-K after NMS
        keep_top_k = 750
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]
        
        # Format outputs as a list of bounding boxes and landmarks
        faces = []
        for i, b in enumerate(dets):
            confidence = float(b[4])
            if confidence < confidence_threshold:
                continue
            box_coords = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
            landmarks_coords = [float(landms[i][j]) for j in range(10)]
            faces.append({
                'box': box_coords,
                'confidence': confidence,
                'landmarks': landmarks_coords
            })
            
        return faces
