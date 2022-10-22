import torch
import pytorch_lightning as pl
from datasets.cbis_ddsm import CBISDDSM
from models.concept_bottleneck_classifier import ConceptBottleneckClassifier
import yaml
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from torchvision.models import resnet18
from nets.concept_classifier import concept_mlp
from torch.utils.data import DataLoader

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

train_loader = DataLoader(data_train, batch_size=32, drop_last=True)
test_loader = DataLoader(data_test, batch_size=32, drop_last=True)

optim_cfg = {'optimizer': torch.optim.Adam, 'lr': 0.0001}
hparams = OmegaConf.create({'lambda_concept': 0.5})

logger = TensorBoardLogger(
        save_dir="./runs-cb", name=experiment_name, default_hp_metric=False
    )

extractor_net = resnet18(pretrained=True)
classifier_net = concept_mlp(33, 2)
model = ConceptBottleneckClassifier(
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
    model=model, train_dataloaders=train_loader, val_dataloaders=test_loader
)