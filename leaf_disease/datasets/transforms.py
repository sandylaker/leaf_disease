from mmcv import Registry, build_from_cfg
from typing import List, Dict, Union, Tuple
import albumentations as A
import albumentations.augmentations.transforms as T
import inspect
import torch
from torch import Tensor
from torchvision.transforms import ToTensor as VisionToTensor
import numpy as np


PIPELINE = Registry('pipeline')


def register_albu_transforms():
    albu_transforms = []
    for module_name in dir(T):
        if module_name.startswith('_'):
            continue
        transform = getattr(T, module_name)
        if inspect.isclass(transform) and issubclass(transform, A.BasicTransform):
            PIPELINE.register_module()(transform)
            albu_transforms.append(module_name)
    return albu_transforms


albu_transforms = register_albu_transforms()


class ToTensor(A.ImageOnlyTransform):
    """Convert image from np.ndarray to torch.Tensor"""
    def __init__(self, dtype=torch.float32):
        super(ToTensor, self).__init__(always_apply=True)
        self.dtype=dtype
        self.to_tensor = VisionToTensor()

    def apply(self, img, **params):
        return self.to_tensor(img).to(self.dtype)

    def get_params(self):
        return {'dtype': self.dtype}


PIPELINE.register_module('ToTensor', module=ToTensor)


def build_pipeline(cfg: Union[Dict, List]):
    if isinstance(cfg, Dict):                               
        return build_from_cfg(cfg, PIPELINE)
    else:
        pipeline = []
        for transform_cfg in cfg:
            t = build_pipeline(transform_cfg)
            pipeline.append(t)
        return A.Compose(pipeline)


class CutMix:
    def __init__(self, num_classes, num_mixes=1, beta=1.0, p=0.6):
        self.num_classes = num_classes
        self.num_mixes = num_mixes
        self.beta = beta
        self.p = p

    def __call__(self,
                 img_src: Tensor,
                 y_src: Union[int, Tensor],
                 imgs_to_mix: Union[Tensor, List[Tensor]],
                 y_to_mix: Union[int, Tensor, List[Tensor], List[int]]):
        if isinstance(imgs_to_mix, List):
            # multiple images to mix
            assert isinstance(y_to_mix, List) and len(imgs_to_mix) == len(y_to_mix) \
                   == self.num_mixes
        else:
            assert not isinstance(y_to_mix, List)
            imgs_to_mix = [imgs_to_mix]
            y_to_mix = [y_to_mix]
        # no matter the image will be mixed or not, the target will be converted to one-hot
        # if the target is already in one-hot format, the convertion will be skipped.
        # e.g. target after mix-up is an array of floats.
        y_src = torch_float_one_hot(y_src, self.num_classes)

        if np.random.rand() < self.p:
            for img_new, y_new in zip(imgs_to_mix, y_to_mix):

                lam = np.random.beta(self.beta, self.beta)
                y_new = torch_float_one_hot(y_new, self.num_classes)
                bbx1, bby1, bbx2, bby2 = self._random_bbox(img_src.shape[-2],
                                                           img_src.shape[-1],
                                                           lam)
                img_src[:, bby1: bby2, bbx1: bbx2] = img_new[:, bby1: bby2, bbx1: bbx2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img_src.shape[-2] * img_src.shape[-1]))
                y_src = y_src * lam + y_new * (1.0 - lam)
        return img_src, y_src

    @staticmethod
    def _random_bbox(height, width, lam):
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(width * cut_rat)
        cut_h = np.int(height * cut_rat)

        cx = np.random.randint(width)
        cy = np.random.randint(height)

        bbx1 = np.clip(cx - cut_w // 2, 0, width)
        bby1 = np.clip(cy - cut_h // 2, 0, height)
        bbx2 = np.clip(cx + cut_w // 2, 0, width)
        bby2 = np.clip(cy + cut_h // 2, 0, height)
        return bbx1, bby1, bbx2, bby2


def torch_float_one_hot(y, num_classes):
    """If y is already a list/tuple/tensor/ndarray of float, just return y directly; otherwise
    convert y to one-hot format. Note that multi-label problem is also supported"""
    if isinstance(y, (List, Tuple, np.ndarray)) and isinstance(y[0], float):
        return y
    elif isinstance(y, Tensor) and (not y.dtype == torch.long):
        return y
    one_hot = torch.zeros(num_classes)
    one_hot[y] = 1.0
    return one_hot


class MixUp:
    def __init__(self, num_classes, beta=1.0, p=0.6, clip_lam=(0.4, 0.6)):
        self.num_classes = num_classes
        self.beta = beta
        self.p = p
        self.clip_lam = clip_lam

    def __call__(self,
                 img_src: Tensor,
                 y_src: Union[int, Tensor],
                 img_to_mix: Tensor,
                 y_to_mix: Union[int, Tensor]):
        # no matter the image will be mixed or not, the target will be converted to one-hot.
        y_src = torch_float_one_hot(y_src, self.num_classes)

        if np.random.rand() < self.p:
            lam = np.random.beta(self.beta, self.beta)
            if self.clip_lam:
                lam = np.clip(lam, a_min=self.clip_lam[0], a_max=self.clip_lam[1])
            img_src = img_src * lam + (1.0 - lam) * img_to_mix
            y_to_mix = torch_float_one_hot(y_to_mix, self.num_classes)
            y_src = y_src * lam + (1.0 - lam) * y_to_mix
        return img_src, y_src