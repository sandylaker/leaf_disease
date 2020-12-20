from .base import BaseWeight
import numpy as np


class PseudoWeight(BaseWeight):
    def __init__(self):
        super(PseudoWeight, self).__init__()

    def _get_cls_weights(self, targets, cls_inds, counts):
        return np.ones_like(cls_inds, dtype=float)