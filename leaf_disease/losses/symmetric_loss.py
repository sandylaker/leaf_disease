import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import weight_reduce_loss


class SCELoss(nn.Module):
    def __init__(self, alpha, beta, clamp_val=1e-4, reduction='mean', loss_weight=1.0):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.clamp_val = clamp_val
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        ce = F.cross_entropy(cls_score, label, reduction='none')

        pred = F.softmax(cls_score, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        one_hot = F.one_hot(label, pred.shape[-1]).to(pred)
        one_hot = torch.clamp(one_hot, min=self.clamp_val, max=1.0)
        rce = - torch.sum(pred * torch.log(one_hot), dim=1)

        loss = self.alpha * ce + self.beta * rce
        return self.loss_weight * weight_reduce_loss(loss, weight, reduction, avg_factor)

