_base_ = ['../_base_/leaf_dataset.py', '../_base_/efficientnet.py', '../_base_/runtime.py']

pretrained = 'pretrained/adv-efficientnet-b5-86493f6b.pth'
model = dict(
    scale=5)

img_size = 456
img_norm_config = dict(
    mean=[0.485, 0.465, 0.406],
    std=[0.229, 0.224, 0.225])

train_pipeline = [
    dict(type='RandomResizedCrop',
         height=img_size,
         width=img_size,
         scale=(0.6, 1.0),
         p=1.0),
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
    dict(type='Normalize',
         always_apply=True,
         **img_norm_config),
    dict(type='Cutout',
         num_holes=8,
         max_h_size=int(img_size*0.2),
         max_w_size=int(img_size*0.2),
         p=0.5),
    dict(type='ToTensor')]

test_pipeline = [
    dict(type='CenterCrop',
         height=img_size,
         width=img_size,
         always_apply=True),
    dict(type='Resize',
         height=img_size,
         width=img_size,
         always_apply=True),
    dict(type='Normalize',
         **img_norm_config),
    dict(type='ToTensor')]


data = dict(
    data_loader=dict(
        batch_size=8,
        shuffle=True,
        num_workers=8,
        timeout=240),
    train=dict(
        pipeline=train_pipeline,
        return_weight=True),
    val=dict(
        pipeline=test_pipeline,
        return_weight=True))