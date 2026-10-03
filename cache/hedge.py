"""
How much the cache trusts each expert.

Every expert starts with the same weight. When a chunk an expert removed
is needed again, that expert loses some weight. This follows LeCaR, with
one change: the penalty is scaled by how relevant the chunk was (its
cross-encoder score), so removing an important chunk costs more.
"""

import numpy as np


class HedgeWeights:

    def __init__(self, n_experts: int = 3, learning_rate: float = 0.45,
                 discount_rate: float = None, capacity: int = 100):
        self.n_experts = n_experts
        self.learning_rate = learning_rate
        # By default a mistake made `capacity` questions ago counts for only
        # 0.5% of a fresh one (the same setting as LeCaR).
        if discount_rate is None:
            discount_rate = 0.005 ** (1.0 / capacity)
        self.discount_rate = discount_rate

        self.W = np.ones(n_experts, dtype=np.float64) / n_experts
        self.n_updates = 0

    def sample(self) -> int:
        """Pick an expert. Usually at random by weight, but once one expert
        has more than 0.6 of the weight, always follow that one."""
        if np.max(self.W) > 0.6:
            return int(np.argmax(self.W))
        return int(np.random.choice(self.n_experts, p=self.W))

    def update(self, expert_idx: int, ce_score: float, evicted_time: int, current_time: int):
        """Penalize an expert whose removed chunk was needed again.

        ce_score is the chunk's relevance from 0 to 1. Mistakes from long
        ago count for less than recent ones.
        """
        questions_since = max(0, current_time - evicted_time)
        penalty = float(ce_score) * self.discount_rate ** questions_since

        self.W[expert_idx] *= np.exp(-self.learning_rate * penalty)

        total = self.W.sum()
        if total > 0:
            self.W /= total

        # Keep every weight between 0.01 and 0.99 so no expert is ever
        # switched off completely.
        self.W = np.clip(self.W, 0.01, 0.99)
        self.W /= self.W.sum()

        self.n_updates += 1

    def get_weights(self) -> np.ndarray:
        return self.W.copy()
