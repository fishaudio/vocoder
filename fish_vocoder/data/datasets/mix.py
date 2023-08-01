import random

from torch.utils.data import IterableDataset


class MixDatast(IterableDataset):
    def __init__(self, datasets: dict[str, dict]):
        values = list(datasets.values())
        probs = [v["prob"] for v in values]
        self.datasets = [v["dataset"] for v in values]

        total_probs = sum(probs)
        self.probs = [p / total_probs for p in probs]

    def __iter__(self):
        while True:
            # Randomly select a dataset
            dataset = random.choices(self.datasets, weights=self.probs)[0]
            data = random.choice(dataset)

            yield data
