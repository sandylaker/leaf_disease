import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import weight_reduce_loss


def mix_cross_entropy(pred,
                      label,
                      weight=None,
                      reduction='mean',
                      avg_factor=None):
    loss = - torch.sum(F.log_softmax(pred, 1) * label, dim=1)

    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


class MixCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MixCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.criterion = mix_cross_entropy

    def forward(self,
                pred,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.criterion(
            pred,
            label,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_cls