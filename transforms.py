import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import Tuple, List, Optional

import torch
from torch import Tensor

try:
    import accimage
except ImportError:
    accimage = None

from torchvision.transforms.transforms import F
from torchvision.transforms.transforms import InterpolationMode, _interpolation_modes_from_int

import numpy as np

__all__ = ["Compose", "ToTensor", "Normalize", "Resize", "Pad", "RandomCrop", "RandomHorizontalFlip", "RandomErasing", "RandomColoring"]

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img, is_rgb):
        for t in self.transforms:
            img, is_rgb = t(img, is_rgb)
        return img, is_rgb
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ToTensor:
    def __call__(self, pic, is_rgb):
        return F.to_tensor(pic), is_rgb
    def __repr__(self):
        return self.__class__.__name__ + '()'

class Normalize(torch.nn.Module):
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace
    def forward(self, tensor: Tensor, is_rgb) -> Tuple[Tensor, int]:
        return F.normalize(tensor, self.mean, self.std, self.inplace), is_rgb
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Resize(torch.nn.Module):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size
        self.max_size = max_size
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img, is_rgb):
        return F.resize(img, self.size, self.interpolation, self.max_size, self.antialias), is_rgb
    def __repr__(self):
        interpolate_str = self.interpolation.value
        return self.__class__.__name__ + '(size={0}, interpolation={1}, max_size={2}, antialias={3})'.format(
            self.size, interpolate_str, self.max_size, self.antialias)

class Pad(torch.nn.Module):
    def __init__(self, padding, fill=0, padding_mode="constant"):
        super().__init__()
        if not isinstance(padding, (numbers.Number, tuple, list)):
            raise TypeError("Got inappropriate padding arg")

        if not isinstance(fill, (numbers.Number, str, tuple)):
            raise TypeError("Got inappropriate fill arg")

        if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
            raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

        if isinstance(padding, Sequence) and len(padding) not in [1, 2, 4]:
            raise ValueError("Padding must be an int or a 1, 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img, is_rgb):
        return F.pad(img, self.padding, self.fill, self.padding_mode), is_rgb

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.padding, self.fill, self.padding_mode)

class RandomCrop(torch.nn.Module):
    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        w, h = F._get_image_size(img)
        th, tw = output_size
        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )
        if w == tw and h == th:
            return 0, 0, h, w
        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return i, j, th, tw
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
    def forward(self, img, is_rgb):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
        width, height = F._get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        i, j, h, w = self.get_params(img, self.size)
        return F.crop(img, i, j, h, w), is_rgb
    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(self.size, self.padding)

class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def forward(self, img, is_rgb):
        if torch.rand(1) < self.p:
            return F.hflip(img), is_rgb
        return img, is_rgb
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomErasing(object):
    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=3.33):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2
    def __call__(self, img, is_rgb):
        if np.random.uniform(0, 1) >= self.p:return img, is_rgb
        mean = [0.4914, 0.4822, 0.4465]#cifar10
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = np.random.uniform(self.sl, self.sh) * area
            #用np.random和测试的random.random分开
            aspect_ratio = np.random.uniform(self.r1, self.r2)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                x1 = np.random.randint(0, img.size()[1] - h)
                y1 = np.random.randint(0, img.size()[2] - w)
                img[0, x1:x1 + h, y1:y1 + w] = mean[0]
                img[1, x1:x1 + h, y1:y1 + w] = mean[1]
                img[2, x1:x1 + h, y1:y1 + w] = mean[2]
                return img, is_rgb
        return img, is_rgb

class RandomColoring(object):
    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=3.33):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2
    def __call__(self, img, is_rgb):
        if np.random.uniform(0, 1) >= self.p:return img, is_rgb
        cvt = RGB_HSV()
        img = cvt.rgb_to_hsv(img.unsqueeze(0))[0]
        for attempt in range(5):
            area = img.size()[1] * img.size()[2]
            target_area = np.random.uniform(self.sl, self.sh) * area
            #用np.random和测试的random.random分开
            aspect_ratio = np.random.uniform(self.r1, self.r2)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                x1 = np.random.randint(0, img.size()[1] - h)
                y1 = np.random.randint(0, img.size()[2] - w)
                if is_rgb == 1:
                    img[0, x1:x1 + h, y1:y1 + w] = np.random.uniform(0, 1)
                    img[1, x1:x1 + h, y1:y1 + w] = img[1, x1:x1 + h, y1:y1 + w] * 0.5 + 0.5 * np.random.uniform(0, 1)# RGB饱和度高，现在要降一些，适应IR
                    img[2, x1:x1 + h, y1:y1 + w] = img[2, x1:x1 + h, y1:y1 + w] * 0.9 + 0.1 * np.random.uniform(1, 1./(img[2, x1:x1 + h, y1:y1 + w].max()))# RGB亮度没有很高，现在要高一些，适应IR
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = np.random.uniform(0, 1) #(img[0, x1:x1 + h, y1:y1 + w] + np.random.uniform(0, 1))%1
                    img[1, x1:x1 + h, y1:y1 + w] = img[1, x1:x1 + h, y1:y1 + w] * 0.5 + 0.5 * np.random.uniform(0, 1)
                    img[2, x1:x1 + h, y1:y1 + w] = img[2, x1:x1 + h, y1:y1 + w] * 0.5 + 0.5 * np.random.uniform(0, 1)
        img = cvt.hsv_to_rgb(img.unsqueeze(0))[0]
        return img, is_rgb

class RandomColoring_Global(object):
    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=3.33):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2
    def __call__(self, img, is_rgb):
        if np.random.uniform(0, 1) >= self.p:return img, is_rgb
        if is_rgb == 1:
            tgt = random.randint(0, 2)
            img[0] = img[tgt]
            img[1] = img[tgt]
            img[2] = img[tgt]
        return img, is_rgb

class RGB_HSV(torch.nn.Module):
    def __init__(self, eps=1e-8):
        super(RGB_HSV, self).__init__()
        self.eps = eps

    def rgb_to_hsv(self, img):

        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

        hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,2]==img.max(1)[0] ]
        hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,1]==img.max(1)[0] ]
        hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,0]==img.max(1)[0] ]) % 6

        hue[img.min(1)[0]==img.max(1)[0]] = 0.0
        hue = hue/6

        saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + self.eps )
        saturation[ img.max(1)[0]==0 ] = 0

        value = img.max(1)[0]
        
        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value],dim=1)
        return hsv

    def hsv_to_rgb(self, hsv):
        h,s,v = hsv[:,0,:,:],hsv[:,1,:,:],hsv[:,2,:,:]
        #对出界值的处理
        h = h%1
        s = torch.clamp(s,0,1)
        v = torch.clamp(v,0,1)
  
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        
        hi = torch.floor(h * 6)
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - (f * s))
        t = v * (1 - ((1 - f) * s))
        
        hi0 = hi==0
        hi1 = hi==1
        hi2 = hi==2
        hi3 = hi==3
        hi4 = hi==4
        hi5 = hi==5
        
        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]
        
        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]
        
        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]
        
        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]
        
        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]
        
        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]
        
        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        return rgb
