from hydra.utils import instantiate
from torch.distributed import get_world_size
from torch.utils.data import Dataset


class RepeatDataset(Dataset):
    def __init__(self, dataset: dict | Dataset, repeat: int | None = None) -> None:
        """Repeat a dataset. Useful for DDP training.

        Args:
            dataset (Union[dict, Dataset]): Dataset to repeat.
            repeat (int, optional): Number of times to repeat, will detect number of GPUs if None. Defaults to None.
            collate_fn (Callable, optional): Collate function. Defaults to None.
        """  # noqa: E501

        self.repeat = repeat

        if isinstance(dataset, dict):
            self.dataset = instantiate(dataset)
        else:
            self.dataset = dataset

    def _get_repeat(self):
        if self.repeat is not None:
            return self.repeat

        return get_world_size()

    def __len__(self):
        return len(self.dataset) * self._get_repeat()

    def __getitem__(self, idx):
        return self.dataset[idx // self._get_repeat()]
