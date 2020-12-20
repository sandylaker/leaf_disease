import torch.nn as nn
from .bce import bce_with_logits
from .cross_entropy import cross_entropy
from mmcv.ops import sigmoid_focal_loss, softmax_focal_loss
from .utils import weight_reduce_loss
from functools import partial


def _sigmoid_focal_loss(pred, target, gamma=2.0, alpha=0.25, weight=None, reduction='none'):
    """wrap sigmoid focal loss, as the function in mmcv does not accept key word argument"""
    return sigmoid_focal_loss(pred, target, gamma, alpha, weight, reduction)


def _softmax_focal_loss(pred, target, gamma=2.0, alpha=0.25, weight=None, reduction='none'):
    return softmax_focal_loss(pred, target, gamma, alpha, weight, reduction)


class CBLoss(nn.Module):
    def __init__(self,
                 loss_type='sigmoid_focal',
                 gamma=2.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(CBLoss, self).__init__()
        assert loss_type in ['bce', 'ce', 'sigmoid_focal', 'softmax_focal']
        self.gamma = None
        self.loss_type = loss_type
        if self.loss_type == 'bce':
            self.criterion = bce_with_logits
        elif self.loss_type == 'ce':
            self.criterion = cross_entropy
        elif self.loss_type == 'sigmoid_focal':
            # set alpha to 0.5, this is equivalent to multiplying the focal loss
            # without alpha with a scale factor 0.5. It is so computed because the focal loss
            # in mmcv does not support array-like alpha.
            self.gamma = gamma
            self.criterion = partial(_sigmoid_focal_loss, gamma=self.gamma, alpha=0.5)
        else:
            self.gamma = gamma
            self.criterion = partial(_softmax_focal_loss, gamma=self.gamma, alpha=0.5)
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if weight is None:
            weight = pred.new_ones((pred.shape[0],))
        else:
            assert weight.ndim == 1

        if self.loss_type in ['sigmoid_focal', 'bce']:
            weight = weight.unsqueeze(1)
            weight = weight.repeat(1, pred.shape[1])

        loss = self.criterion(pred, target, weight=None, reduction='none')
        if self.loss_type in ['sigmoid_focal', 'softmax_focal']:
            # rescale the loss, since alpha is set to 0.5
            loss = 2 * loss
        loss = self.loss_weight * weight_reduce_loss(loss, weight, reduction)
        return loss