import torch.nn as nn
import torch.nn.functional as F
from .utils import weight_reduce_loss
import torch


def bce_with_logits(pred, label, weight=None, reduction='mean', avg_factor=None):
    # element-wise losses
    if label.shape != pred.shape:
        label = F.one_hot(label, num_classes=pred.shape[-1]).to(torch.float32)
    loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
        if weight.ndim != loss.ndim:
            weight = weight.unsqueeze(-1)
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


class BCEWithLogits(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(BCEWithLogits, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.cls_criterion = bce_with_logits

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls