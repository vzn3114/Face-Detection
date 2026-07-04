import os
import sys
import torch

# Add backend directory to sys.path so inner imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from backend.data.config import cfg_mnet, cfg_re50
from backend.models.retinaface import RetinaFace
from backend.segmentation.model import UNet

def remove_prefix(state_dict, prefix):
    print("remove prefix '{}'".format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_retinaface_model(backbone_name):
    if backbone_name == "mobile0.25":
        cfg = cfg_mnet.copy()
        trained_model = './backend/weights/mobilenet0.25_Final.pth'
    else:
        cfg = cfg_re50.copy()
        trained_model = './backend/weights/Resnet50_Final.pth'
        
    cfg['pretrain'] = False  # Avoid loading nonexistent pretrain tar file for inference/export
    net = RetinaFace(cfg=cfg, phase='test')
    
    pretrained_dict = torch.load(trained_model, map_location='cpu')
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    
    net.load_state_dict(pretrained_dict, strict=False)
    net.eval()
    return net


def export_retinaface(backbone_name, output_path):
    print(f"Loading RetinaFace with {backbone_name} backbone...")
    model = load_retinaface_model(backbone_name)
    
    # RetinaFace supports dynamic dimensions for input
    dummy_input = torch.randn(1, 3, 640, 640)
    print(f"Exporting RetinaFace ({backbone_name}) to ONNX format at {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['boxes', 'confs', 'landmarks'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'boxes': {0: 'batch_size', 1: 'num_anchors'},
            'confs': {0: 'batch_size', 1: 'num_anchors'},
            'landmarks': {0: 'batch_size', 1: 'num_anchors'}
        },
        opset_version=11
    )
    print(f"RetinaFace ({backbone_name}) exported successfully.")

def export_unet(output_path):
    print("Loading U-Net...")
    model = UNet(n_channels=3, n_classes=19)
    trained_model = './backend/weights/unet_face_celeb.pth'
    
    model.load_state_dict(torch.load(trained_model, map_location='cpu'))
    model.eval()
    
    dummy_input = torch.randn(1, 3, 512, 512)
    print(f"Exporting U-Net to ONNX format at {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=11
    )
    print("U-Net exported successfully.")

if __name__ == "__main__":
    os.makedirs('./backend/weights', exist_ok=True)
    try:
        export_retinaface('mobile0.25', './backend/weights/mobilenet0.25_Final.onnx')
    except Exception as e:
        print(f"Error exporting RetinaFace (MobileNet0.25): {e}")
        
    try:
        export_retinaface('resnet50', './backend/weights/Resnet50_Final.onnx')
    except Exception as e:
        print(f"Error exporting RetinaFace (ResNet50): {e}")
        
    try:
        export_unet('./backend/weights/unet_face_celeb.onnx')
    except Exception as e:
        print(f"Error exporting U-Net: {e}")
