from typing import Any, Dict, Sequence, Tuple, Union

import hydra
import pytorch_lightning as pl
import torch
import torchmetrics
from omegaconf import DictConfig
from torch.optim import Optimizer
import torch.nn.functional as F
from torch import nn


class ConceptBottleneckClassifier(pl.LightningModule):
    def __init__(
        self,
        extractor_net: torch.nn.Module,
        classifier_net: torch.nn.Module,
        train_mode,
        optim_cfg,
        hparams: DictConfig,
        num_classes,
        use_sigmoid = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cfg = hparams
        self.save_hyperparameters(hparams)

        self.optim_cfg = optim_cfg
        self.extractor_net = extractor_net
        self.classifier_net = classifier_net
        self.extractor_net.conv1 = nn.Conv2d(
            1,
            self.extractor_net.conv1.out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.use_sigmoid = use_sigmoid

        train_mode_list = ['joint', 'only_classifier', 'only_extractor', 'joint_no_concepts', 'classifier_from_gt_concepts']
        if train_mode not in train_mode_list:
            raise ValueError(f'train_mode must be one of {train_mode_list} but was {train_mode}')

        self.train_mode = train_mode

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.train_concept_accuracy = torchmetrics.Accuracy()
        self.val_concept_accuracy = torchmetrics.Accuracy()
        self.test_concept_accuracy = torchmetrics.Accuracy()
        self.train_AUC = torchmetrics.AUROC(num_classes=num_classes)
        self.val_AUC = torchmetrics.AUROC(num_classes=num_classes)
        self.test_AUC = torchmetrics.AUROC(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier_net(self.extractor_net(x).sigmoid())

    def training_step(self, batch: Any, batch_idx: int):
        x, con, y = batch
        concept_logits = self.extractor_net(x)
        concept_preds = concept_logits.sigmoid()
        logits = self.classifier_net(concept_preds if self.use_sigmoid else concept_logits)
        with torch.no_grad():
            pred = torch.softmax(logits, dim=-1)
            self.train_accuracy(pred, y)
            self.train_AUC(pred, y)
            self.train_concept_accuracy(concept_preds, con)
        class_loss = F.cross_entropy(logits, y)
        concept_loss = F.binary_cross_entropy_with_logits(concept_logits, con.float())

        loss = 0
        if self.train_mode in ['joint', 'only_classifier', 'joint_no_concepts', 'classifier_from_gt_concepts']:
            loss += class_loss
        if self.train_mode in ['joint', 'only_extractor']:
            loss += self.cfg.lambda_concept * concept_loss

        self.log("loss/train", loss, batch_size=len(y))
        self.log("class_loss/train", class_loss, batch_size=len(y), prog_bar=True)
        self.log("concept_loss/train", concept_loss, batch_size=len(y), prog_bar=True)
        self.log_dict(
            {
                "accuracy/train": self.train_accuracy,
                "AUC/train": self.train_AUC,
                "concept_accuracy/train": self.train_concept_accuracy,
            },
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x, con, y = batch
        concept_logits = self.extractor_net(x)
        concept_preds = concept_logits.sigmoid()
        logits = self.classifier_net(concept_preds if self.use_sigmoid else concept_logits)
        pred = torch.softmax(logits, dim=-1)
        self.val_accuracy(pred, y)
        self.val_AUC(pred, y)
        self.val_concept_accuracy(concept_preds, con)
        class_loss = F.cross_entropy(logits, y)
        concept_loss = F.binary_cross_entropy_with_logits(concept_logits, con.float())

        loss = 0
        if self.train_mode in ['joint', 'only_classifier', 'joint_no_concepts', 'classifier_from_gt_concepts']:
            loss += class_loss
        if self.train_mode in ['joint', 'only_extractor']:
            loss += self.cfg.lambda_concept * concept_loss

        self.log("loss/val", loss, batch_size=len(y))
        self.log("class_loss/val", class_loss, batch_size=len(y), prog_bar=True)
        self.log("concept_loss/val", concept_loss, batch_size=len(y), prog_bar=True)
        self.log_dict(
            {
                "accuracy/val": self.val_accuracy,
                "AUC/val": self.val_AUC,
                "concept_accuracy/val": self.val_concept_accuracy,
            },
            prog_bar=True,
        )

    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch
        logits = self.forward(x)
        pred = torch.softmax(logits, dim=-1)
        self.test_accuracy(pred, y)
        self.test_AUC(pred, y)
        loss = F.cross_entropy(logits, y)

        self.log("loss/test", loss, batch_size=len(y))
        self.log_dict(
            {
                "accuracy/test": self.test_accuracy,
                "AUC/test": self.test_AUC,
            },
            prog_bar=True,
        )

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:

        outer_optimizer = self.optim_cfg['optimizer'](params=self.parameters(), lr=self.optim_cfg['lr'])
        return outer_optimizer
