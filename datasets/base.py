import itertools
import random
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Iterator

import torch
from torch.utils.data import Dataset

from utils.collate_fn import identity
from .task import TaskTarget, Task


class TaskSource(Dataset, ABC):
    task_target: TaskTarget
    labels: Sequence
    classes: Sequence = None


class MetaDataset(Dataset, ABC):
    @abstractmethod
    def __getitem__(self, index) -> Task:
        ...


class SubtaskMetaDataset(MetaDataset):
    def __init__(
        self, dataset: TaskSource, collate_fn=None, n_shot_provider:Iterator[Sequence[int]]=None
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.task_target = dataset.task_target

        self.indices = None

        if self.task_target is TaskTarget.MULTICLASS_CLASSIFICATION:
            self.indices = {
                c: [i for i, l in enumerate(self.dataset.labels) if l == c]
                for c in self.dataset.classes
            }
        elif self.task_target is TaskTarget.MULTILABEL_CLASSIFICATION:
            self.indices = {
                c: [i for i, l in enumerate(self.dataset.labels) if l[c]]
                for c in self.dataset.classes
            }
        else:
            raise NotImplementedError

        self.collate_fn = collate_fn if collate_fn else identity

        self.n_shot_provider = (
            n_shot_provider if n_shot_provider else itertools.repeat((5, None))
        )

    def sample_multiclass_shot(
        self,
        selected_classes,
        n_shot_per_class: Optional[int] = None,
        n_shot_total: Optional[int] = None,
    ):
        if n_shot_per_class is None:
            if n_shot_total is None:
                raise ValueError(
                    "n_shot_per_class and n_shot_total can not both be None."
                )
            population = [
                (i, j) for j, c in enumerate(selected_classes) for i in self.indices[c]
            ]
            sample = random.sample(population, n_shot_total)
        else:
            if n_shot_total is not None:
                raise ValueError(
                    "n_shot_per_class and n_shot_total can not both be set."
                )
            sample = [
                (i, j)
                for j, c in enumerate(selected_classes)
                for i in random.sample(self.indices[c], n_shot_per_class)
            ]
        return [(self.dataset[i][0], torch.tensor(j)) for i, j in sample]

    def __getitem__(self, index) -> Task:
        if self.task_target is TaskTarget.MULTICLASS_CLASSIFICATION:
            support = self.collate_fn(
                self.sample_multiclass_shot(index, *next(self.n_shot_provider))
            )
            query = self.collate_fn(
                self.sample_multiclass_shot(index, *next(self.n_shot_provider))
            )
        return Task(support, query, self.task_target)
