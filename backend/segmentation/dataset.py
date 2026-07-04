import os
import random
import numpy as np
from PIL import Image
import PIL.ImageEnhance as ImageEnhance
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from segmentation.config import IMAGE_DIR, MASK_DIR, INPUT_SIZE

# ==============================================================================
# Augmentations (Copied from Project B and optimized)
# ==============================================================================

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        if (W, H) == (w, h):
            return dict(im=im, lb=lb)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            lb = lb.resize((w, h), Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return dict(
            im=im.crop(crop),
            lb=lb.crop(crop)
        )


class HorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']

            # Attributes mapping in CelebAMask-HQ:
            # 2: l_brow, 3: r_brow
            # 4: l_eye,  5: r_eye
            # 7: l_ear,  8: r_ear
            # Swapping these indices on horizontal flip
            flip_lb = np.array(lb)
            l_brow = (lb == 2)
            r_brow = (lb == 3)
            l_eye = (lb == 4)
            r_eye = (lb == 5)
            l_ear = (lb == 7)
            r_ear = (lb == 8)

            flip_lb[l_brow] = 3
            flip_lb[r_brow] = 2
            flip_lb[l_eye] = 5
            flip_lb[r_eye] = 4
            flip_lb[l_ear] = 8
            flip_lb[r_ear] = 7

            flip_lb = Image.fromarray(flip_lb)
            return dict(
                im=im.transpose(Image.FLIP_LEFT_RIGHT),
                lb=flip_lb.transpose(Image.FLIP_LEFT_RIGHT)
            )


class RandomScale(object):
    def __init__(self, scales=(1,)):
        self.scales = scales

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        W, H = im.size
        scale = random.choice(self.scales)
        w, h = int(W * scale), int(H * scale)
        return dict(
            im=im.resize((w, h), Image.BILINEAR),
            lb=lb.resize((w, h), Image.NEAREST)
        )


class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None):
        if brightness is not None and brightness > 0:
            self.brightness = [max(1 - brightness, 0), 1 + brightness]
        if contrast is not None and contrast > 0:
            self.contrast = [max(1 - contrast, 0), 1 + contrast]
        if saturation is not None and saturation > 0:
            self.saturation = [max(1 - saturation, 0), 1 + saturation]

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)
        return dict(im=im, lb=lb)


class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb


# ==============================================================================
# FaceMask Dataset Class (Refactored and model-independent)
# ==============================================================================

class FaceMask(Dataset):
    def __init__(self, rootpth=None, cropsize=(448, 448), mode='train'):
        super().__init__()
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.ignore_lb = 255
        
        # Determine paths dynamically
        if rootpth is None:
            self.image_dir = IMAGE_DIR
            self.mask_dir = MASK_DIR
        else:
            self.image_dir = os.path.join(rootpth, 'CelebA-HQ-img')
            self.mask_dir = os.path.join(rootpth, 'mask')

        # Filter out non-image files and sort them numerically
        self.imgs = sorted(
            [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
            key=lambda x: int(os.path.splitext(x)[0])
        )

        # PyTorch Image Normalization (ImageNet standards)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # Data augmentation for training mode
        self.trans_train = Compose([
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            HorizontalFlip(),
            RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomCrop(cropsize)
        ])

    def __getitem__(self, idx):
        impth = self.imgs[idx]
        img = Image.open(os.path.join(self.image_dir, impth))
        img = img.resize(INPUT_SIZE, Image.BILINEAR)
        
        # Load preprocessed mask
        mask_filename = os.path.splitext(impth)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_filename)
        
        # Load mask and convert to 'P' (Palette/Grayscale) mode
        label = Image.open(mask_path).convert('P')

        if self.mode == 'train':
            im_lb = dict(im=img, lb=label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
            
        img = self.to_tensor(img)
        # Add channel dimension: (1, H, W) and cast to int64 for CrossEntropyLoss
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        return img, label

    def __len__(self):
        return len(self.imgs)
