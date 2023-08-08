import os
import random

import numpy as np
import torch
from torch.distributed import get_rank, is_initialized
from torch.utils.data import IterableDataset


class MixDatast(IterableDataset):
    def __init__(self, datasets: dict[str, dict]):
        values = list(datasets.values())
        probs = [v["prob"] for v in values]
        self.datasets = [v["dataset"] for v in values]

        total_probs = sum(probs)
        self.probs = [p / total_probs for p in probs]

    def __iter__(self):
        rank = get_rank() if is_initialized() else 0
        seed = (42 + rank * 114 + os.getpid() * 514) % 2**32

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        while True:
            # Randomly select a dataset
            dataset = random.choices(self.datasets, weights=self.probs)[0]
            data = random.choice(dataset)

            yield data
