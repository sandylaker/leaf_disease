from leaf_disease.datasets import build_dataset
import numpy as np
import matplotlib.pyplot as plt


def get_data_config():
    dataset_type = 'LeafDiseaseDataset'
    data_root = 'data/'

    img_norm_config = dict(
        mean=[0.485, 0.465, 0.406],
        std=[0.229, 0.224, 0.225])
    mixup_cfg = dict(p=1)
    cutmix_cfg = dict(num_mixes=1, p=0)

    train_pipeline = [
        dict(type='RandomResizedCrop',
             height=224,
             width=224,
             scale=(0.6, 1.0),
             ratio=(0.75, 1.3)),
        dict(type='HorizontalFlip', p=0.5),
        dict(type='VerticalFlip', p=0.5),
        dict(type='LongestMaxSize',
             max_size=224,
             interpolation=1,
             always_apply=True),
        dict(type='Normalize',
             always_apply=True,
             **img_norm_config),
        dict(type='PadIfNeeded',
             min_height=224,
             min_width=224,
             border_mode=0,
             value=0,
             always_apply=True),
        dict(type='ToTensor')]
    data = dict(
        data_loader=dict(
            batch_size=32,
            shuffle=True,
            num_workers=8,
            timeout=240),
        train_1=dict(
            type=dataset_type,
            img_dir=data_root + 'train_images/',
            annot_file=data_root + 'train.csv',
            indices_file=data_root + 'kfold/train_1.txt',
            pipeline=train_pipeline,
            mixup_cfg=mixup_cfg,
            cutmix_cfg=cutmix_cfg,
            return_weight=True))
    return data


def test_dataset():
    data_cfg = get_data_config()
    train_set= build_dataset(data_cfg['train_1'])
    ind = np.random.choice(len(train_set))

    img, y, w = train_set[ind]
    print(f'img.shape: {img.shape}, img_1.mean: {img.mean()}, img_1.std: {img.std()}')
    print('y:', y)
    print('w:', w)

    img_norm_config = dict(
        mean=[0.485, 0.465, 0.406],
        std=[0.229, 0.224, 0.225])
    plt.imshow(((img.permute(1, 2, 0).numpy() * img_norm_config['std'] +
                 img_norm_config['mean']) * 255.0).astype(np.uint8))
    plt.show()


if __name__ == '__main__':
    test_dataset()