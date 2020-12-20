import pytorch_lightning as pl
from ..models import build_model
from ..losses import build_loss
from .output_transforms import build_output_transform
from mmcv.runner import build_optimizer, obj_from_dict
from mmcv import get_logger
from torch import optim
import torch.nn as nn


class LitModel(pl.LightningModule):
    def __init__(self, cfg, **kwargs):
        super(LitModel, self).__init__(**kwargs)
        self.cfg = cfg
        self.mmcv_logger = get_logger('cassava')

        self.model = build_model(self.cfg.model)
        self.out_transforms_dict = self.build_output_transforms()
        self.criterion = self.build_loss()

        self.metrics = nn.ModuleDict(dict(
            accuracy=pl.metrics.Accuracy(
                compute_on_step=False,
                dist_sync_on_step=True)))

    def build_output_transforms(self):
        out_trans_cfg = self.cfg.output_transforms
        out_transforms_dict = {k: build_output_transform(v) for k, v in out_trans_cfg.items()}
        return out_transforms_dict

    def build_loss(self):
        loss_cfg = self.cfg.loss
        return build_loss(loss_cfg)

    def configure_optimizers(self):
        opt_cfg = self.cfg.optimizer
        optimizer = build_optimizer(self.model, opt_cfg)
        lr_cfg = self.cfg.lr_config
        lr_scheduler = obj_from_dict(lr_cfg,
                                     parent=optim.lr_scheduler,
                                     default_args=dict(
                                         optimizer=optimizer))
        scheduler = dict(
            scheduler=lr_scheduler,
            interval='epoch',
            strict=True,
            name='lr')
        return [optimizer], [scheduler]
        
    def training_step(self, batch, batch_idx):
        img, target, weight = batch
        pred = self.model.forward(img)
        loss = self.criterion(pred, target, weight=weight)
        self.log('train_loss',
                 loss,
                 prog_bar=False,
                 logger=True,
                 on_step=True,
                 on_epoch=True,
                 sync_dist=True)
        return dict(loss=loss)

    def validation_step(self, batch, batch_idx):
        img, target, weight = batch
        pred = self.model.forward(img)
        loss = self.criterion(pred, target, weight=weight)

        self.log('val_loss',
                 loss,
                 prog_bar=True,
                 logger=True,
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True)

        return pred, target, weight

    def validation_step_end(self, val_step_out):
        pred, target, weight = val_step_out
        for name, metric in self.metrics.items():
            pred, target = self.out_transforms_dict[name](pred, target, weight)
            metric.update(pred, target)

    def validation_epoch_end(self, outputs):
        for name, metric in self.metrics.items():
            value = metric.compute()
            self.log(
                name,
                value,
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True)
