from mmcv import Registry, build_from_cfg
from .pseudo import PseudoWeight
from .inverse_frequency import IFWeight
from .class_balanced import CBWeight

WEIGHTS = Registry('weights')

WEIGHTS.register_module(module=PseudoWeight)
WEIGHTS.register_module(module=IFWeight)
WEIGHTS.register_module(module=CBWeight)


def build_weights(cfg, default_args=None):
    return build_from_cfg(cfg, WEIGHTS, default_args=default_args)