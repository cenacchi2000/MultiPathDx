import math
from typing import Iterable, List, Tuple


class LogisticRegression:
    """Simple binary logistic regression implemented from scratch."""

    def __init__(self, n_features: int, lr: float = 0.01):
        self.n_features = n_features
        self.lr = lr
        self.weights = [0.0 for _ in range(n_features)]
        self.bias = 0.0

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def predict_proba(self, features: List[float]) -> float:
        z = sum(w * f for w, f in zip(self.weights, features)) + self.bias
        return self._sigmoid(z)

    def predict(self, features: List[float]) -> int:
        return 1 if self.predict_proba(features) >= 0.5 else 0

    def fit(self, data: Iterable[Tuple[List[float], int]], epochs: int = 10) -> None:
        for _ in range(epochs):
            for features, label in data:
                pred = self.predict_proba(features)
                error = pred - label
                for i in range(self.n_features):
                    self.weights[i] -= self.lr * error * features[i]
                self.bias -= self.lr * error
