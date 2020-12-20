_base_ = './b2_512_ce.py'

loss = dict(
    _delete_=True,
    type='LabelSmoothLoss',
    label_smooth_val=0.1,
    num_classes=5,
)