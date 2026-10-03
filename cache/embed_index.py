"""
Holds the embeddings of the cached chunks in one fixed-size numpy matrix,
so a lookup is a single matrix multiply. The cache is small, so this is
fast enough without FAISS or a GPU.
"""

import numpy as np


class EmbedIndex:

    def __init__(self, capacity: int, dim: int):
        self.capacity = capacity
        self.dim = dim
        self._matrix = np.zeros((capacity, dim), dtype=np.float32)
        self._free_slots = list(range(capacity))
        self._doc_to_slot = {}

    @property
    def size(self) -> int:
        return len(self._doc_to_slot)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._doc_to_slot

    def add(self, doc_id: str, embedding: np.ndarray):
        """The embedding must already be normalized."""
        if doc_id in self._doc_to_slot:
            raise ValueError(f"doc_id '{doc_id}' is already in the index")
        if not self._free_slots:
            raise RuntimeError("index is full, evict something first")

        slot = self._free_slots.pop(0)
        self._matrix[slot] = embedding.astype(np.float32)
        self._doc_to_slot[doc_id] = slot

    def remove(self, doc_id: str):
        if doc_id not in self._doc_to_slot:
            raise KeyError(f"doc_id '{doc_id}' is not in the index")

        slot = self._doc_to_slot.pop(doc_id)
        self._matrix[slot] = 0.0
        self._free_slots.append(slot)

    def search(self, query: np.ndarray, top_k: int = 10, threshold: float = 0.5) -> list:
        """Returns up to top_k (doc_id, similarity) pairs at or above the
        threshold, best first. The query must already be normalized."""
        if not self._doc_to_slot:
            return []

        doc_ids, doc_matrix = self.get_doc_matrix()
        sims = doc_matrix @ query.astype(np.float32)

        k = min(top_k, len(doc_ids))
        best = np.argpartition(sims, -k)[-k:]
        best = best[np.argsort(sims[best])[::-1]]
        return [(doc_ids[i], float(sims[i])) for i in best if float(sims[i]) >= threshold]

    def get_doc_matrix(self) -> tuple:
        """Returns (doc_ids, embeddings) for every cached chunk, or ([], None)."""
        if not self._doc_to_slot:
            return [], None

        doc_ids = list(self._doc_to_slot.keys())
        slots = [self._doc_to_slot[d] for d in doc_ids]
        return doc_ids, self._matrix[slots]
