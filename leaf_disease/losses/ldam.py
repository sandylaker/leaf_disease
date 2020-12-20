import torch.nn as nn
from .cross_entropy import cross_entropy
import numpy as np
import torch


class LDAMLoss(nn.Module):
    def __init__(self,
                 train_cls_num_list,
                 test_cls_num_list=None,
                 max_m=0.5,
                 s=30,
                 reduction='mean',
                 loss_weight=None):
        super(LDAMLoss, self).__init__()
        if test_cls_num_list is not None:
            test_cls_num_list = np.asarray(test_cls_num_list)
            train_cls_num_list = np.asarray(train_cls_num_list)
            m_list = np.sqrt(np.sqrt(test_cls_num_list / train_cls_num_list))
        else:
            m_list = 1.0 / np.sqrt(np.sqrt(train_cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.tensor(m_list, dtype=torch.float32)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.reduction = reduction
        if loss_weight is None:
            loss_weight = 1.0
        self.loss_weight = loss_weight

    def to(self, *args, **kwargs):
        super(LDAMLoss, self).to(*args, **kwargs)
        self.m_list = self.m_list.to(*args, **kwargs)
        return self

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                weight=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        index = torch.zeros_like(pred, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.to(pred)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        pred_m = pred - batch_m

        output = torch.where(index, pred_m, pred)
        loss_cls = self.loss_weight * cross_entropy(self.s * output,
                                                    target,
                                                    weight=weight,
                                                    reduction=reduction)
        return loss_cls
