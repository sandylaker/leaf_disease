_base_ = './b2_512_ce.py'

model = dict(
    normed_fc=dict(
        tau=1.0, 
        norm_input=True))

loss = dict(
    type='LDAMLoss',
    train_cls_num_list=[869, 1751, 1909, 10527, 2061],
    test_cls_num_list=[218, 438, 477, 2631, 516],
    max_m=0.3,
    s=30,
    loss_weight=1.0)

max_epochs = 5
# at first stage, keep learning rate constant
lr_config = dict(
    _delete_=True,
    type='StepLR',
    step_size=max_epochs,
    gamma=1.0)


