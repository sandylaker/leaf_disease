from abc import ABCMeta, abstractmethod
import numpy as np


class BaseWeight(metaclass=ABCMeta):

    def __init__(self):
        pass

    def get_weights(self, targets, key='index', idx_to_class: dict = None) -> dict:
        assert key in ['index', 'name']
        if key == 'name':
            assert idx_to_class is not None
        targets = np.asarray(targets)
        assert targets.ndim == 1, 'Targets must be a flattened array.'

        cls_inds, counts = np.unique(targets, return_counts=True)
        cls_weights = self._get_cls_weights(targets, cls_inds, counts)
        if key == 'name':
            weights_per_class = self.convert_key(cls_inds, cls_weights, idx_to_class)
        else:
            weights_per_class = dict(zip(cls_inds, cls_weights))
        return weights_per_class

    @abstractmethod
    def _get_cls_weights(self, targets, cls_inds, counts):
        raise NotImplementedError

    @staticmethod
    def convert_key(cls_inds, cls_weights, idx_to_class: dict) -> dict:
        weights_per_class = {idx_to_class[cls_ind]: cls_weight for cls_ind, cls_weight in zip(
            cls_inds, cls_weights)}
        return weights_per_class