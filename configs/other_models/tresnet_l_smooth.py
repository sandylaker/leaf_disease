_base_ = ['./tresnet_l.py']

loss = dict(
    _delete_=True,
    type='LabelSmoothLoss',
    label_smooth_val=0.1,
    num_classes=5,
)

optimizer = dict(
    weight_decay=3e-6)