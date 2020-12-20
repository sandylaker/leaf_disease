from leaf_disease.models import build_model
import timm.models as models


def test_build_models():
    cfg = dict(
        type='seresnext50_32x4d',
        num_classes=5,
        pretrained=True)

    model = build_model(cfg)
    assert isinstance(model, models.resnet.ResNet)