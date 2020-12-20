_base_ = './b2_512_ce.py'

data = dict(
    train=dict(
        mixup_cfg=dict(p=0),
        cutmix_cfg=dict(p=0.5)),
    val=dict(
        mixup_cfg=dict(p=0),
        cutmix_cfg=dict(p=0)))

loss = dict(
    type='MixCrossEntropyLoss')

output_transforms = dict(
    loss=dict(
        type='InactivatedTransform'),
    accuracy=dict(
        type='OneHotToIndicesTransform'))