from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os.path as osp


def kfold(train_csv, save_dir, **kwargs):
    df = pd.read_csv(train_csv)
    num_total = df.shape[0]
    kf = StratifiedKFold(**kwargs)
    x = np.zeros(num_total)
    y = df['label'].values

    for i, (train_inds, val_inds) in enumerate(kf.split(x, y)):
        train_file = osp.join(save_dir, f'train_{i+1}.txt')
        val_file = osp.join(save_dir, f'val_{i+1}.txt')

        np.savetxt(train_file, train_inds, fmt='%d')
        np.savetxt(val_file, val_inds, fmt='%d')
        print(f'Fold {i+1} is saved in directory {save_dir}')