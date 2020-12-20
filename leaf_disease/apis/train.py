from .lit_model import LitModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from mmcv.runner import set_random_seed
import os
import os.path as osp


def train_classifier(cfg,
                     train_loader,
                     val_loader,
                     work_dir=os.getcwd(),
                     seed=2020,
                     **kwargs):
    set_random_seed(seed, deterministic=True)
    lit_model = LitModel(cfg)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=cfg.callbacks['patience'],
                                   mode='min')
    model_checkpoint = ModelCheckpoint(filepath=None,
                                       monitor='accuracy',
                                       verbose=True,
                                       save_last=True,
                                       save_top_k=2,
                                       mode='max',
                                       dirpath=work_dir + '/checkpoints/',
                                       filename='{epoch}-{accuracy:.5f}')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    tb_name = osp.splitext(osp.basename(cfg.filename))[0]
    tb_logger = TensorBoardLogger(save_dir=work_dir,
                                  name=tb_name)
    kwargs.update(dict(
        logger=tb_logger,
        callbacks=[early_stopping,
                   model_checkpoint,
                   lr_monitor]))
    trainer = pl.Trainer(**kwargs)
    trainer.fit(lit_model, train_loader, val_loader)
