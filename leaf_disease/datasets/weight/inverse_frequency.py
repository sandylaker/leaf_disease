from .base import BaseWeight
import numpy as np


class IFWeight(BaseWeight):
    """Weight the classes by inverse of frequency"""
    def __init__(self):
        super(IFWeight, self).__init__()

    def _get_cls_weights(self, targets, cls_inds, counts):
        return self.weight_by_freq(targets.shape[0], cls_inds.shape[0], counts)

    @staticmethod
    def weight_by_freq(num_samples: int, num_classes: int, counts: np.ndarray):
        cls_weights = 1 / counts
        scale = num_samples / num_classes
        cls_weights *= scale
        return cls_weights