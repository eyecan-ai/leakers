from typing import Sequence
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class GaloisDataset:
    def __init__(self, dataset: Sequence, expansion: int = 1) -> None:
        self._dataset = dataset
        self._expansion = expansion
        self._size = len(self._dataset)
        self._expanded_size = self._size * self._expansion

    def __len__(self) -> int:
        return self._expanded_size

    def __getitem__(self, index: int) -> Sequence:
        return self._dataset[index % self._size]


class GenericAlphabetDatamodule(pl.LightningDataModule):
    def __init__(
        self, dataset, batch_size: int, drop_last: bool = False, exapansion: int = 1
    ):
        super().__init__()
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._dataset = dataset
        self._expansion = exapansion

    def train_dataloader(self):
        return DataLoader(
            GaloisDataset(dataset=self._dataset, expansion=self._expansion),
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=self._drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self._dataset,
            batch_size=self._batch_size,
            shuffle=False,
            drop_last=self._drop_last,
        )

    def test_dataloader(self):
        return DataLoader(
            self._dataset,
            batch_size=self._batch_size,
            shuffle=False,
            drop_last=self._drop_last,
        )
