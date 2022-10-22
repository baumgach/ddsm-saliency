import torch
import pytorch_lightning as pl
from datasets.cbis_ddsm import CBISDDSM
from models import concept_bottleneck_classifier
import yaml
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from torchvision.models import resnet18
from nets.concept_classifier import concept_mlp

def load_yaml(path):

    with open(path, 'r') as stream:
        try:
            parsed_yaml=yaml.safe_load(stream)
            return parsed_yaml
        except yaml.YAMLError as exc:
            print(exc)


experiment_name = 'test'

data_root = '/mnt/qb/work/baumgartner/cbaumgartner/CBIS-DDSM'
data_train = CBISDDSM(root_path=data_root, with_concepts=True)
data_test = CBISDDSM(root_path=data_root, with_concepts=True, split='test')

optim_cfg = OmegaConf.create(load_yaml('conf/optim/default.yaml'))
hparams = OmegaConf.create({'lambda_concept': 0.5})

logger = TensorBoardLogger(
        save_dir="./runs-cb", name=experiment_name, default_hp_metric=False
    )

breakpoint()
extractor_net = resnet18(pretrained=True)
classifier_net = concept_mlp(33, 2)
model = concept_bottleneck_classifier(
    extractor_net=extractor_net,
    classifier_net=classifier_net,
    train_mode='joint',
    hparams=hparams,
    optim_cfg=optim_cfg,
    num_classes=2,
)

trainer = pl.Trainer(
    logger=logger,
    val_check_interval=0.25,
    log_every_n_steps=50,
    accelerator="gpu",
    devices=1,
)
trainer.fit(
    model=model, train_dataloaders=data_train, val_dataloaders=data_test
)