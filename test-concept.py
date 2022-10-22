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

model = ModelClass.load_from_checkpoint(checkpoint)

for ii, data in enumerate(data_test):

    x, c, y = data
    c_p = model.extractor_net(x)

    print(ii, c, y, c_p)