from torch import nn


def concept_mlp(num_concepts, num_classes):
    return nn.Sequential(
        nn.Linear(num_concepts, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, num_classes),
        nn.ReLU(inplace=True),
    )
