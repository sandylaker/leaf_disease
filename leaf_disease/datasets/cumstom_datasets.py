import os.path as osp
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from mmcv import list_from_file
from leaf_disease.datasets.transforms import build_pipeline, MixUp, CutMix
from leaf_disease.datasets.weight import build_weights
import cv2


class LeafDiseaseDataset(Dataset):

    idx_to_class = {0: "Cassava Bacterial Blight (CBB)",
                    1: "Cassava Brown Streak Disease (CBSD)",
                    2: "Cassava Green Mottle (CGM)",
                    3: "Cassava Mosaic Disease (CMD)",
                    4: "Healthy"}

    def __init__(self,
                 img_dir,
                 annot_file,
                 indices_file,
                 pipeline,
                 mixed_label=False,
                 mixup_cfg=None,
                 cutmix_cfg=None,
                 weight_cfg=None,
                 return_weight=False):
        super(LeafDiseaseDataset, self).__init__()
        self.img_dir = img_dir
        self.annot_file = annot_file
        self.indices_file = indices_file
        self.pipeline = build_pipeline(pipeline)
        self.mixed_label = mixed_label
        self.mixup_cfg = mixup_cfg
        self.cutmix_cfg = cutmix_cfg

        annot_df = pd.read_csv(annot_file).astype({'label': 'int32'})
        indices = list_from_file(indices_file)
        indices = [int(ind) for ind in indices]

        # indices of data frame after indexing are not continuous, reset them to maintain continuity
        self.annot_df = annot_df.loc[indices].reset_index(drop=True)
        if not self.mixed_label:
            self.targets = self.annot_df['label'].values
        else:
            self.mixed_label_cols = [f'mixed_{i}' for i in self.idx_to_class.keys()]
            self.targets = self.annot_df[self.mixed_label_cols].values

        if weight_cfg is None:
            weight_cfg = dict(type='PseudoWeight')
        self.weight_calc = build_weights(weight_cfg)
        self.cls_weights = self.weight_calc.get_weights(
            self.annot_df['label'].values, key='index', idx_to_class=None)

        self.return_weight = return_weight
        if self.return_weight:
            self.sample_weights = self.convert_weights(self.cls_weights,
                                                       self.annot_df['label'].values)
            assert len(self.annot_df) == len(self.sample_weights)

        # build mixup and cutmix transformations
        self.mixup = None
        self.cutmix = None
        if self.mixup_cfg:
            self.mixup = MixUp(num_classes=len(self.cls_weights), **self.mixup_cfg)
        if self.cutmix_cfg:
            self.cutmix = CutMix(num_classes=len(self.cls_weights), **self.cutmix_cfg)

    def __len__(self):
        return len(self.annot_df)

    def __getitem__(self, index):
        img, target = self._pre_pipeline(index)
        img = self.pipeline(image=img)['image']
        if self.mixup is not None:
            ind_to_mix = np.random.choice(len(self.targets))
            img_to_mix, target_to_mix = self._pre_pipeline(ind_to_mix)
            img_to_mix = self.pipeline(image=img_to_mix)['image']
            img, target = self.mixup(img, target, img_to_mix, target_to_mix)
        if self.cutmix is not None:
            img_to_mix_list = []
            target_to_mix_list = []
            for i in range(self.cutmix.num_mixes):
                ind_to_mix = np.random.choice(len(self.targets))
                img_to_mix, target_to_mix = self._pre_pipeline(ind_to_mix)
                img_to_mix = self.pipeline(image=img_to_mix)['image']
                img_to_mix_list.append(img_to_mix)
                target_to_mix_list.append(target_to_mix)
            img, target = self.cutmix(img, target, img_to_mix_list, target_to_mix_list)

        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32)

        weight = None
        if self.return_weight:
            weight = self.sample_weights[index]
        return img, target, weight

    def _pre_pipeline(self, index):
        img_name = self.annot_df.loc[index, 'image_id']
        if not self.mixed_label:
            target = int(self.annot_df.loc[index, 'label'])
        else:
            target = self.targets[index]
        img_file = osp.join(self.img_dir, img_name)

        img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
        return img, target

    @staticmethod
    def convert_weights(cls_weights: dict, targets):
        targets = np.asarray(targets)
        weights_per_sample = np.zeros_like(targets, dtype=float)
        for cls_ind, cls_w in cls_weights.items():
            weights_per_sample[targets == cls_ind] = cls_w
        return list(weights_per_sample)