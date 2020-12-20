import torch
import torch.nn as nn
import torch.nn.functional as F
from leaf_disease.losses import CrossEntropyLoss, MixCrossEntropyLoss, LDAMLoss, BiTemperedLoss
import numpy as np


def test_mce():
    ce = CrossEntropyLoss()
    mce = MixCrossEntropyLoss()

    pred = torch.randn(10, 5)
    y = torch.randint(0, 5, (10,))
    y_onehot = torch.zeros_like(pred).scatter(1, y.unsqueeze(1), 1)
    weight = torch.rand(y.shape)

    loss_ce = ce(pred, y, weight=weight)
    loss_mce = mce(pred, y_onehot, weight=weight)
    assert torch.allclose(loss_ce, loss_mce)


def test_ldam():
    class LDAMLossOfficial(nn.Module):
        # official implementation
        def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
            super(LDAMLossOfficial, self).__init__()
            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float32)
            self.m_list = m_list
            assert s > 0
            self.s = s
            self.weight = weight

        def forward(self, x, target):
            index = torch.zeros_like(x, dtype=torch.uint8)
            index.scatter_(1, target.data.view(-1, 1), 1)

            index_float = index.type_as(x)
            batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
            batch_m = batch_m.view((-1, 1))
            x_m = x - batch_m

            output = torch.where(index, x_m, x)
            return F.cross_entropy(self.s * output, target, weight=self.weight)

    pred = torch.randn(5, 5)
    y = torch.randint(0, 5, (5,))
    cls_num_list = [869, 1751, 1909, 10527, 2061]
    max_m = 0.5
    s = 30
    ldam_ours = LDAMLoss(train_cls_num_list=cls_num_list, max_m=max_m, s=s)
    ldam_official = LDAMLossOfficial(cls_num_list, max_m=max_m, s=s)

    with torch.no_grad():
        loss_ours = ldam_ours(pred, y)
        loss_offical = ldam_official(pred, y)
        assert torch.allclose(loss_ours, loss_offical)


def test_bi_tempered():
    bit_criterion = BiTemperedLoss(1.0, 1.0, 0.0, 5)
    ce_criterion = CrossEntropyLoss()

    pred = torch.randn(10, 5)
    label = torch.randint(0, 5, (10, ))

    loss_bit = bit_criterion(pred, label)
    loss_ce = ce_criterion(pred, label)
    assert torch.allclose(loss_bit, loss_ce), f'loss_bit: {loss_bit}; \nloss_ce: {loss_ce}'



if __name__ == '__main__':
    test_bi_tempered()