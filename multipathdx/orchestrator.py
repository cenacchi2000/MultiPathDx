from typing import List
from .model import LogisticRegression
from .dataset import EDAICDataset


class MultiPathDx:
    """A minimal multi-pathology screening system."""

    def __init__(self, dataset: EDAICDataset, lr: float = 0.01, epochs: int = 10):
        self.dataset = dataset
        self.model = LogisticRegression(dataset.feature_dim(), lr=lr)
        self.epochs = epochs

    def train(self) -> None:
        data = [(s.features, s.label) for s in self.dataset.samples]
        self.model.fit(data, epochs=self.epochs)

    def evaluate(self) -> float:
        correct = 0
        total = 0
        for sample in self.dataset.samples:
            pred = self.model.predict(sample.features)
            if pred == sample.label:
                correct += 1
            total += 1
        return correct / total if total else 0.0
