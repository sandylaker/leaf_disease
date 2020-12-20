from mmcv.utils import build_from_cfg, Registry
from .efficientnet import EfficientNet
from timm import create_model
from copy import deepcopy

MODELS = Registry('models')
MODELS.register_module('EfficientNet', module=EfficientNet)


def build_model(cfg, default_args=None):
    if cfg['type'] not in MODELS:
        return buil_model_from_timm(cfg, default_args=default_args)
    else:
        return build_from_cfg(cfg, MODELS, default_args)


def buil_model_from_timm(cfg, default_args=None):
    cfg = deepcopy(cfg)
    model_name = cfg.pop('type')
    cfg.update({'model_name': model_name})
    if default_args is not None:
        cfg.update(**default_args)
    return create_model(**cfg)