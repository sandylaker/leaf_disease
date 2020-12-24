from mmcv import Registry, build_from_cfg
from .cumstom_datasets import LeafDiseaseDataset

DATASETS = Registry('datasets')
DATASETS.register_module(name='LeafDiseaseDataset', module=LeafDiseaseDataset)


def build_dataset(cfg, default_args=None):
    return build_from_cfg(cfg, registry=DATASETS, default_args=default_args)