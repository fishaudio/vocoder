from typing import Dict, Optional

import lightning as L
from torch.utils.data import DataLoader, Dataset, IterableDataset


class NaiveDataModule(L.LightningDataModule):
    def __init__(
        self,
        datasets: Dict[str, Dataset],
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        drop_last: bool = False,
        persistent_workers: bool = True,
        collate_fn: Optional[callable] = None,
        train_batch_size: Optional[int] = None,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
    ):
        super().__init__()

        self.splits: dict[str, Dataset] = datasets
        self.batch_size = batch_size
        self.train_batch_size = train_batch_size or batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.test_batch_size = test_batch_size or batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers
        self.collate_fn = collate_fn

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.splits["train"],
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn,
            shuffle=not isinstance(self.splits["train"], IterableDataset),
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        return DataLoader(
            self.splits["val"],
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        # This probably won't be used

        assert self.batch_size == 1, "Batch size must be 1 for test set"

        return DataLoader(
            self.splits["test"],
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
        )
