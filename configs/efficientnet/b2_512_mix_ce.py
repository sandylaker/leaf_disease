_base_ = './b2_512_ce.py'

data_root = 'data/'
data = dict(
    train=dict(
        mixed_label=True,
        annot_file=data_root + '/kfold/train_mixed.csv',
        indices_file=data_root + 'kfold/train_5.txt',
        return_weight=False),
    val=dict(
        mixed_label=True,
        annot_file=data_root + '/kfold/train_mixed.csv',
        indices_file=data_root + 'kfold/val_5.txt',
        return_weight=False))

loss = dict(
    type='MixCrossEntropyLoss')

output_transforms = dict(
    accuracy=dict(
        type='OneHotToIndicesTransform'))