from mmcv import Registry, build_from_cfg
from .cumstom_datasets import LeafDiseaseDataset, UnsupervisedDataset

DATASETS = Registry('datasets')
DATASETS.register_module(name='LeafDiseaseDataset', module=LeafDiseaseDataset)
DATASETS.register_module(name='UnsupervisedDataset', module=UnsupervisedDataset)


def build_dataset(cfg, default_args=None):
    return build_from_cfg(cfg, registry=DATASETS, default_args=default_args)