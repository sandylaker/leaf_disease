import torch
import torch.nn.functional as F
from .base import BaseTransform


class ActivatedTransform(BaseTransform):
    """
    Tranform:
        pred to sigmoid(pred) or softmax(pred)
        y to y or one_hot(y)
    """
    def __init__(self, is_multilabel=False, round=False, return_weight=False):
        super(ActivatedTransform, self).__init__()
        self.is_multilabel=is_multilabel
        self.round = round
        self.return_weight = return_weight

    def _transform(self, y_pred, y, weight):
        # transform y_pred
        if self.is_multilabel or y_pred.ndim == 1:
            y_pred = torch.sigmoid(y_pred)
            if self.round:
                y_pred = y_pred.round()
        else:
            y_pred = torch.softmax(y_pred, dim=-1)
        # transform y
        if self.is_multilabel:
            y = F.one_hot(y, num_classes=y_pred.shape[-1])

        if self.return_weight:
            return y_pred, y, dict(weight=weight)
        else:
            return y_pred, y