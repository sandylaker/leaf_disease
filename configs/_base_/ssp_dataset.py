dataset_type = 'UnsupervisedDataset'
data_root = 'data/'

img_norm_config = dict(
    mean=[0.485, 0.465, 0.406],
    std=[0.229, 0.224, 0.225])

img_size = 224
train_pipeline = [
    dict(type='ShiftScaleRotate',
         shift_limit=0.2,
         scale_limit=0.2,
         rotate_limit=15,
         border_mode=0,
         p=0.6),
    dict(type='Flip', p=0.5),
    dict(type='RandomRotate90',
         p=0.5),
    dict(type='RandomBrightness',
         limit=0.2,
         p=0.6),
    dict(type='RandomContrast',
         limit=0.2,
         p=0.6),
    dict(type='HueSaturationValue',
         hue_shift_limit=20,
         sat_shift_limit=20,
         val_shift_limit=20,
         p=0.6),
    dict(type='RandomResizedCrop',
         height=img_size,
         width=img_size,
         scale=(0.9, 1.0),
         ratio=(0.75, 1.3333),
         always_apply=True),
    dict(type='Normalize',
         always_apply=True,
         **img_norm_config),
    dict(type='ToTensor')]

test_pipeline = [
    dict(type='CenterCrop',
         height=224,
         width=224,
         always_apply=True),
    dict(type='Normalize',
         **img_norm_config),
    dict(type='ToTensor')]

data = dict(
    data_loader=dict(
        batch_size=32,
        shuffle=True,
        num_workers=8,
        timeout=240),
    train=dict(
        type=dataset_type,
        img_dir=data_root + 'train_images/',
        annot_file=data_root + 'train.csv',
        indices_file=data_root + 'kfold/train_1.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_dir=data_root + 'train_images/',
        annot_file=data_root + 'train.csv',
        indices_file=data_root + 'kfold/val_1.txt',
        pipeline=test_pipeline))


