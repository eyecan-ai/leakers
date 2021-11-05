import pytorch_lightning as pl
from torch.utils.data import DataLoader


class GenericAlphabetDatamodule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size: int, drop_last: bool = False):
        super().__init__()
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._dataset = dataset

    def train_dataloader(self):
        return DataLoader(
            self._dataset,
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
