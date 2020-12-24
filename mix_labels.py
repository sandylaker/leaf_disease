import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from leaf_disease import build_dataset, build_model, collate_fn
from copy import deepcopy
import os.path as osp
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
from mmcv import Config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('checkpoint', type=str, help='path to check point')
    parser.add_argument('--indices-file',
                        type=str,
                        help='path to txt file that stores the indices of validation samples')
    parser.add_argument('--save-path',
                        type=str,
                        help='path for saving file',
                        default='data/kfold/mix_label.csv')
    parser.add_argument('--alpha',
                        type=float,
                        help='proportion of truth label',
                        default=0.7)
    parser.add_argument('--num-classes',
                        type=int,
                        help='number of classes (for one-hot encoding)',
                        default=5)
    parser.add_argument('--num-tta',
                        type=int,
                        help='number of tta',
                        default=5)
    parser.add_argument('--device',
                        type=str,
                        help='device name',
                        default='cuda:0')
    args = parser.parse_args()
    return args


def get_tta_pipeline(cfg):
    pipeline = A.Compose([
        A.RandomResizedCrop(
            cfg['img_size'],
            cfg['img_size'],
            scale=(0.5, 1.0),
            always_apply=True),
        A.Transpose(p=0.5),
        A.Flip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.6),
        A.HueSaturationValue(
            hue_shift_limit=0.2,
            sat_shift_limit=0.2,
            val_shift_limit=0.2,
            p=0.6),
        A.Normalize(always_apply=True,
                    **cfg['img_norm_config']),
        ToTensorV2(always_apply=True)])
    return pipeline


def prepare_data(cfg, indices_file=None):
    dataset_cfg = deepcopy(cfg.data['val'])
    if indices_file is not None:
        dataset_cfg.update({'indices_file': indices_file})
    val_set = build_dataset(dataset_cfg)
    val_annot_df = val_set.annot_df
    data_loader_cfg = deepcopy(cfg.data['data_loader'])
    data_loader_cfg.update({'batch_size': 4 * data_loader_cfg['batch_size'],
                            'shuffle': False})
    val_loader = DataLoader(dataset=val_set,
                            collate_fn=collate_fn,
                            **data_loader_cfg)
    print(f'len(val_loader): {len(val_loader)}')
    return val_loader, val_annot_df


class TTAEngine:
    def __init__(self,
                 cfg,
                 check_point,
                 indices_file,
                 save_path,
                 num_tta,
                 alpha=0.7,
                 num_classes=5,
                 device='cuda:0'):
        assert not osp.isdir(save_path), f'save_path should be a file name, but got {save_path}'
        self.cfg = cfg
        self.save_path = save_path
        self.indices_file = indices_file
        self.num_tta = num_tta
        self.alpha = alpha
        self.device = device
        self.num_classes = num_classes
        self.val_loader, self.val_annot_df = prepare_data(cfg, self.indices_file)

        self.model = build_model(cfg.model)
        check_point = torch.load(check_point)
        unmatched_keys = self.model.load_state_dict(check_point)
        print(unmatched_keys)
        self.model = self.model.to(device)
        self.model.eval()

    def predict_and_mix(self):
        preds = self.predict_tta()
        assert preds.shape[0] == len(self.val_annot_df)
        assert preds.ndim == 1

        self.val_annot_df['pseudo_label'] = preds
        self.val_annot_df = self.val_annot_df.astype({'pseudo_label': int})
        mixed_label = self.mix(self.val_annot_df['label'],
                         self.val_annot_df['pseudo_label'],
                         self.num_classes,
                         self.alpha)
        mixed_cols = [f'mixed_{i}' for i in range(self.num_classes)]
        self.val_annot_df[mixed_cols] = mixed_label
        self.val_annot_df.to_csv(self.save_path)

    @staticmethod
    def mix(truth_label, pseudo_label, num_classes=5, alpha=0.7):
        assert len(truth_label) == len(pseudo_label)
        truth_label = truth_label.astype(int)
        pseudo_label = pseudo_label.astype(int)
        N = truth_label.shape[0]
        truth_onehot = np.zeros((N, num_classes))
        truth_onehot[np.arange(N), truth_label] = 1

        pseudo_onehot = np.zeros((N, num_classes))
        pseudo_onehot[np.arange(N), pseudo_label] = 1
        mixed = truth_onehot * alpha + (1 - alpha) * pseudo_onehot
        assert np.allclose(mixed.sum(1), np.ones(N))
        return mixed

    def predict_tta(self):
        preds = [self.predict_single() for _ in range(self.num_tta)]
        preds = np.stack(preds, axis=0).mean(0).argmax(1)
        return preds

    def predict_single(self):
        preds = []
        with torch.no_grad():
            for batch in tqdm(self.val_loader):
                img, target, _ = batch
                img = img.to(self.device)

                pred = self.model(img)
                preds.append(pred.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        return preds


def main():
    args = vars(parse_args())
    cfg = Config.fromfile(args.pop('config'))
    tta_engine = TTAEngine(cfg,
                           check_point=args['checkpoint'],
                           indices_file=args.get('indices_file', None),
                           save_path=args['save_path'],
                           num_tta=args['num_tta'],
                           num_classes=args['num_classes'],
                           alpha=args['alpha'],
                           device=args['device'])
    tta_engine.predict_and_mix()


if __name__ == '__main__':
    main()