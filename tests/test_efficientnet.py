from leaf_disease.models.efficientnet import EfficientNet
from leaf_disease.models import NormedLinear
import torch
import torch.nn.functional as F


def test_efficiennet_ns():
    model = EfficientNet(frozen_stages=8)
    assert not model.head[0].weight.requires_grad
    assert model.head[6].weight.requires_grad
    model = EfficientNet(frozen_stages=9)
    assert not model.head[0].weight.requires_grad
    assert not model.head[6].weight.requires_grad


def test_normed_linear():
    x = torch.randn(10, 5)
    layer = NormedLinear(5, 3, tau=1.0, norm_input=True)
    x_true = F.normalize(x, dim=1).mm(F.normalize(layer.weight, dim=0))
    x_layer = layer.forward(x)
    assert torch.allclose(x_true, x_layer)
                    

if __name__ == '__main__':
    test_normed_linear()