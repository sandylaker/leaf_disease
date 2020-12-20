from mmcv.utils import Registry, build_from_cfg
from .focal import FocalLoss
from .cross_entropy import CrossEntropyLoss
from .bce import BCEWithLogits
from .class_balanced import CBLoss
from .ldam import LDAMLoss
from .mix_cross_entropy import MixCrossEntropyLoss
from .label_smooth_loss import LabelSmoothLoss
from .bi_tempered import BiTemperedLoss
from .symmetric_loss import SCELoss

LOSSES = Registry('losses')
LOSSES.register_module('FocalLoss', module=FocalLoss)
LOSSES.register_module('CrossEntropyLoss', module=CrossEntropyLoss)
LOSSES.register_module('BCEWithLogits', module=BCEWithLogits)
LOSSES.register_module('CBLoss', module=CBLoss)
LOSSES.register_module('LDAMLoss', module=LDAMLoss)
LOSSES.register_module('MixCrossEntropyLoss', module=MixCrossEntropyLoss)
LOSSES.register_module('LabelSmoothLoss', module=LabelSmoothLoss)
LOSSES.register_module('BiTemperedLoss', module=BiTemperedLoss)
LOSSES.register_module('SCELoss', module=SCELoss)


def build_loss(cfg: dict, default_args=None):
    return build_from_cfg(cfg, LOSSES, default_args)