import sys
sys.path.append('..')
from leaf_disease.apis import train_classifier, collate_fn
from leaf_disease.datasets import build_dataset
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from mmcv import Config, get_logger
import argparse
import os.path as osp
from copy import deepcopy


def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--seed', type=int, default=2020, help='seed for initialize the training')

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args = vars(args)

    cfg = Config.fromfile(args.pop('config'))
    work_dir = args.pop('work_dir')
    seed = args.pop('seed')

    # update commandline args with cfg.trainer
    args.update(cfg.trainer)
    mmcv_logger = get_logger('cassava')

    mmcv_logger.info(cfg.pretty_text + '\n' + '-' * 60)
    cfg.dump(osp.join(work_dir, osp.basename(cfg.filename)))

    train_set = build_dataset(cfg.data['train'])
    val_set = build_dataset(cfg.data['val'])
    if train_set.return_weight:
        log_str = '; '.join([f'{k}: {v}' for k, v in train_set.cls_weights.items()])
        mmcv_logger.info(f'Class weights: {log_str}')

    cfg_train_loader = deepcopy(cfg.data['data_loader'])
    train_loader = DataLoader(dataset=train_set,
                              collate_fn=collate_fn,
                              **cfg_train_loader)
    cfg_val_loader = deepcopy(cfg.data['data_loader'])
    cfg_val_loader.update({'batch_size': 4 * cfg_val_loader['batch_size'],
                           'shuffle': False})
    val_loader = DataLoader(dataset=val_set,
                            collate_fn=collate_fn,
                            **cfg_val_loader)

    train_classifier(cfg, train_loader, val_loader, work_dir, seed, **args)
    

if __name__ == '__main__':
    main()