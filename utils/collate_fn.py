import torch


def identity(x):
    return x


def stack(t):
    return tuple(torch.stack(el) for el in zip(*t))
