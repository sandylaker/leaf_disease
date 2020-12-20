from mmcv import Registry, build_from_cfg
from .activated import ActivatedTransform
from .inactivated import InactivatedTransform, OneHotToIndicesTransform


OUTPUT_TRANSFORMS = Registry('output_transforms')
OUTPUT_TRANSFORMS.register_module(name='ActivatedTransform', module=ActivatedTransform)
OUTPUT_TRANSFORMS.register_module(name='InactivatedTransform', module=InactivatedTransform)
OUTPUT_TRANSFORMS.register_module(name='OneHotToIndicesTransform', module=OneHotToIndicesTransform)


def build_output_transform(cfg, default_args=None):
    return build_from_cfg(cfg, OUTPUT_TRANSFORMS, default_args=default_args)