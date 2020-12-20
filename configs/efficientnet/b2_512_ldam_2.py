_base_ = './b2_512_ldam_1.py'

pretrained = 'workdirs/b2_512_ldam_rw/fold_5/model_5_acc=0.8834.pt'

lr = 1e-5
optimizer = dict(
    _delete_=True,
    type='Adam',
    lr=lr,
    weight_decay=1e-6)

max_epochs = 8
lr_config = dict(
    type='StepLR',
    step_size=4,
    gamma=0.5)

weight_cfg = dict(
    type='CBWeight',
    beta=0.99,
    factor=1.0)

data = dict(
    train=dict(
        weight_cfg=weight_cfg,
        return_weight=True),
    val=dict(
        weight_cfg=weight_cfg,
        return_weight=True))