from .base import BaseWeight
import numpy as np


class CBWeight(BaseWeight):
    """Class balanced weights.
    `https://arxiv.org/abs/1901.05555`.
    Note that the class weights are normalized such that theirs sum is equal to
    `factor * num_classes`.

    Args:
        beta (float): parameter beta of CB loss.
        factor(float): a parameter controlling the scale of the sum of weights, which will
            be normalized to `factor * num_classes`.

    """
    def __init__(self, beta=0.9999, factor=1.0):
        super(CBWeight, self).__init__()
        assert beta > 0 and factor > 0, 'beta and factor must be positive'
        self.beta = beta
        self.factor = factor

    def _get_cls_weights(self, targets, cls_inds, counts: np.ndarray):
        num_classes = cls_inds.shape[0]
        w = (1 - self.beta) / (1 - self.beta ** counts + 1e-8)
        scale = self.factor * num_classes / w.sum()
        weights_per_class = w * scale
        return weights_per_class