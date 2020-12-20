_base_ = ['../_base_/leaf_dataset.py',
          '../_base_/efficientnet.py',
          '../_base_/runtime.py']

pretrained = 'pretrained/tf_efficientnet_b2_ns-00306e48.pth'
model = dict(
    pretrained=pretrained,
    scale=2)

img_size = 512
img_norm_config = dict(
    mean=[0.485, 0.465, 0.406],
    std=[0.229, 0.224, 0.225])

train_pipeline = [
    dict(type='RandomResizedCrop',
         height=img_size,
         width=img_size,
         p=1.0),
    dict(type='Transpose',
         p=0.5),
    dict(type='HorizontalFlip',
         p=0.5),
    dict(type='VerticalFlip',
         p=0.5),
    dict(type='ShiftScaleRotate',
         p=0.5),
    dict(type='HueSaturationValue',
         hue_shift_limit=0.2,
         sat_shift_limit=0.2,
         val_shift_limit=0.2,
         p=0.5),
    dict(type='RandomBrightnessContrast',
         brightness_limit=(-0.1, 0.1),
         contrast_limit=(-0.1, 0.1),
         p=0.5),
    dict(type='Normalize',
         always_apply=True,
         **img_norm_config),
    dict(type='CoarseDropout',
         p=0.5),
    dict(type='Cutout',
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
        batch_size=16),
    train=dict(
        pipeline=train_pipeline,
        return_weight=False),
    val=dict(
        pipeline=test_pipeline,
        return_weight=False))