from typing import Any, Optional

import torch
from torch.utils.data import Dataset, RandomSampler, Sampler


class TaskDataset(Dataset):
    def __init__(
        self, datasets: list[Dataset], sampler: Optional[Sampler] = None
    ) -> None:
        self.datasets = datasets
        self.sampler = sampler if sampler else RandomSampler
        super().__init__()

    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)
