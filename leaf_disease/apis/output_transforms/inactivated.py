import torch.nn.functional as F
import torch
from .base import BaseTransform


class InactivatedTransform(BaseTransform):
    """
    Transform:
        pred to pred
        y to y or one_hot(y)
    """
    def __init__(self, is_multilabel=False, return_weight=False):
        super(InactivatedTransform, self).__init__()
        self.is_multilabel = is_multilabel
        self.return_weight = return_weight

    def _transform(self, y_pred, y, weight):
        if self.is_multilabel:
            y = F.one_hot(y, num_classes=y_pred.shape[-1])
        # make the third element a dict, in order to be compatible with API of
        # ignite.metrics.Loss
        if self.return_weight:
            return y_pred, y, dict(weight=weight)
        else:
            return y_pred, y


class OneHotToIndicesTransform(BaseTransform):
    """
    Transform:
        pred to pred
        onehot_y (or discrete distributed y) to y
    """
    def __init__(self, return_weight=False):
        super(OneHotToIndicesTransform, self).__init__()
        self.return_weight = return_weight

    def _transform(self, y_pred, y, weight):
        y = torch.argmax(y, dim=1)
        if self.return_weight:
            return y_pred, y, dict(weight=weight)
        else:
            return y_pred, y
