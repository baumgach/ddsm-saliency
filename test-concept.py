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
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

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

model = ConceptBottleneckClassifier.load_from_checkpoint(
    checkpoint, 
    train_mode='joint',
    hparams=hparams,
    num_classes=2,
    num_concepts=33,
)

# SPICULATED - 32
# OBSCURED - 28
# ILL-DEFINED - 22

concept_dict = {32: 'spiculated', 28: 'obscured', 22: 'ill-defined'}

from captum.attr import IntegratedGradients
for ii, data in tqdm(enumerate(data_test)):

    x, c, y = data
    x = x.unsqueeze(0)
    c_p = torch.sigmoid(model.extractor_net(x))
    ig = IntegratedGradients(model.extractor_net)

    fig = plt.figure()
    plt.imshow(x.detach().numpy().squeeze(), cmap='gray')
    plt.axis('off')
    plt.savefig(f'example_images/img-{str(ii).zfill(3)}-input')

    for k, v in concept_dict.items():

        attr, delta = ig.attribute(x, target=k, return_convergence_delta=True)

        gt_c = c.detach().numpy().squeeze()[k]
        pred_c = np.round(c_p.detach().numpy().squeeze()[k])
        
        fig = plt.figure()
        plt.imshow(x.detach().numpy().squeeze(), cmap='gray')
        plt.imshow(attr.detach().numpy().squeeze(), alpha=0.7, cmap='hot')
        plt.axis('off')
        plt.savefig(f'example_images/img-{str(ii).zfill(3)}-{v}-pred={pred_c}-gt={gt_c}')
        
    if ii > 10:
        break

