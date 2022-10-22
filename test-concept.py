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
import subprocess

checkpoint = '/mnt/qb/work/baumgartner/cbaumgartner/ddsm-saliency/runs-cb/0f5bb7e/version_0/checkpoints/best-auc-epoch=19-step=789.ckpt'
data_root = '/mnt/qb/work/baumgartner/cbaumgartner/CBIS-DDSM'

data_test = CBISDDSM(root_path=data_root, with_concepts=True, split='test')

extractor_net = resnet18(pretrained=True)
classifier_net = concept_mlp(33, 2)
# model = ConceptBottleneckClassifier(
#     extractor_net=extractor_net,
#     classifier_net=classifier_net,
#     train_mode='joint',
#     hparams=hparams,
#     optim_cfg=optim_cfg,
#     num_classes=2,
#     num_concepts=33,
# )
optim_cfg = {'optimizer': torch.optim.Adam, 'lr': 0.0001}
hparams = OmegaConf.create({'lambda_concept': 0.5})

breakpoint()
model = ConceptBottleneckClassifier.load_from_checkpoint(
    checkpoint, 
    train_mode='joint',
    hparams=hparams,
    optim_cfg=optim_cfg,
    num_classes=2,
    num_concepts=33,
)

for ii, data in enumerate(data_test):

    x, c, y = data
    c_p = model.extractor_net(x)

    print(ii, c, y, c_p)