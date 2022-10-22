from typing import Callable

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler

from utils.collate_fn import identity


class DefaultDataModule(LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        batch_size: DictConfig,
        num_workers: DictConfig = None,
    ):
        super().__init__()

        self.datasets = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return self.make_dataloader(
            self.datasets.train, self.batch_size.train, self.num_workers.train
        )

    def val_dataloader(self):
        return self.make_dataloader(
            self.datasets.val, self.batch_size.val, self.num_workers.val
        )

    def test_dataloader(self):
        return self.make_dataloader(
            self.datasets.test, self.batch_size.test, self.num_workers.test
        )

    def predict_dataloader(self):
        pass

    def make_dataloader(self, dataset, batch_size, num_workers):

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )


class MetaDataModule(DefaultDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        batch_size: DictConfig,
        sampler: Callable[[Dataset], Sampler],
        num_workers: DictConfig = None,
    ):
        super().__init__(datasets, batch_size, num_workers)
        self.sampler = sampler

    def make_dataloader(self, dataset, batch_size, num_workers):

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self.sampler(dataset.dataset),
            num_workers=num_workers,
            collate_fn=identity,
            pin_memory=True,
            drop_last=True,
        )
