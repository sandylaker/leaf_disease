_base_ = './b2_512_ce.py'

loss = dict(
    _delete_=True,
    type='BiTemperedLoss',
    num_classes=5,
    t1=0.8,
    t2=1.2)