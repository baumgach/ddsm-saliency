from typing import Sequence, Iterator

import numpy as np
import torch
from torch.utils.data import Sampler

from datasets.base import TaskSource


class RandomMultiClassSubTaskSampler(Sampler[Sequence[int]]):
    def __init__(self, data_source: TaskSource, num_samples: int, n_way_min: int = 2, n_way_max: int = 50):
        self.classes = data_source.classes
        self.num_samples = num_samples
        self.n_way_min = n_way_min
        self.n_way_max = min(n_way_max, len(self.classes))

    def __iter__(self) -> Iterator[Sequence[int]]:
        n_way = np.random.randint(self.n_way_min, self.n_way_max + 1, self.num_samples)
        for n in n_way:
            yield np.random.choice(self.classes, n, replace=False)

    def __len__(self) -> int:
        return self.num_samples
