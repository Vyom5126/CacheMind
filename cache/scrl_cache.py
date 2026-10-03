"""
The hot cache. It keeps a small number of chunk embeddings in memory and
learns which of three eviction experts to trust (see experts.py and
hedge.py). The design follows LeCaR, but works on embeddings instead of
exact keys.
"""

import numpy as np

from cache.embed_index import EmbedIndex
from cache.entry import SCRLEntry
from cache.experts import EXPERT_NAMES, N_EXPERTS, get_all_nominees
from cache.hedge import HedgeWeights
from cache.history import HistoryQueues
from cache.structures import DequeDict, HeapDict


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-10 else v.astype(np.float32)


class SCRLCache:

    def __init__(self, capacity: int, dim: int = 384, learning_rate: float = 0.45,
                 discount_rate: float = None, window_size: int = 50,
                 threshold: float = 0.5, hit_threshold: int = 7):
        self.capacity = capacity
        self.dim = dim
        self.threshold = threshold          # min similarity for a chunk to count as a match
        self.hit_threshold = hit_threshold  # min number of matches for a cache hit
        self.window_size = window_size
        self.time = 0                       # number of questions seen so far

        self.lru = DequeDict()
        self.lfu = HeapDict()
        self.embed_index = EmbedIndex(capacity, dim)

        self.hedge = HedgeWeights(n_experts=N_EXPERTS, learning_rate=learning_rate,
                                  discount_rate=discount_rate, capacity=capacity)
        self.history = HistoryQueues(hist_size=capacity)

        self._q_window = []     # embeddings of recent questions
        self._ret_window = []   # embeddings of chunks the cold search returned recently

        self.hits = 0
        self.misses = 0
        self._expert_evictions = np.zeros(N_EXPERTS, dtype=np.int64)
        self._recent_evictions = []   # emptied by pop_evictions()

    def _add(self, doc_id: str, emb: np.ndarray, freq: int = 1):
        entry = SCRLEntry(doc_id, freq=freq, time=self.time)
        self.lru[doc_id] = entry
        self.lfu[doc_id] = entry
        self.embed_index.add(doc_id, emb)

    def _remove(self, doc_id: str):
        del self.lru[doc_id]
        del self.lfu[doc_id]
        self.embed_index.remove(doc_id)

    def _evict_one(self):
        nominees = get_all_nominees(self.embed_index, self.lfu, self._q_window, self._ret_window)

        expert_idx = self.hedge.sample()
        evict_id = nominees[expert_idx]

        if evict_id is None:
            for i, nominee in enumerate(nominees):
                if nominee is not None:
                    evict_id, expert_idx = nominee, i
                    break
        if evict_id is None:
            return

        # If every expert picked the same chunk, nobody gets blamed for it later.
        if len({n for n in nominees if n is not None}) == 1:
            expert_idx = -1

        entry = self.lru[evict_id]
        entry.evicted_time = self.time
        self._remove(evict_id)
        self.history.add(entry, expert_idx)
        self._recent_evictions.append(evict_id)

        if expert_idx >= 0:
            self._expert_evictions[expert_idx] += 1

    def _hit_update(self, doc_id: str):
        entry = self.lru[doc_id]
        entry.freq += 1
        entry.time = self.time
        # Setting them again moves the entry to the newest end of the LRU
        # list and to its new place in the LFU heap.
        self.lru[doc_id] = entry
        self.lfu[doc_id] = entry

    def request(self, query: np.ndarray) -> list:
        """Look up a normalized question embedding. Returns the matching
        (doc_id, similarity) pairs. It is a hit if there are at least
        hit_threshold of them."""
        self.time += 1

        self._q_window.append(query.astype(np.float32))
        if len(self._q_window) > self.window_size:
            self._q_window.pop(0)

        if self.embed_index.size == 0:
            self.misses += 1
            return []

        results = self.embed_index.search(query, top_k=10, threshold=self.threshold)

        if len(results) >= self.hit_threshold:
            self.hits += 1
            for doc_id, _ in results:
                if doc_id in self.lru:
                    self._hit_update(doc_id)
        else:
            self.misses += 1

        return results

    def insert(self, doc_id: str, emb: np.ndarray, ce_score: float = 1.0):
        """Add a chunk from the cold search, removing one first if the cache
        is full. ce_score is its relevance, used if it was removed by mistake
        earlier."""
        if doc_id in self.lru:
            self._hit_update(doc_id)
            return

        # If this chunk was removed recently, the expert that removed it made
        # a mistake. Penalize that expert and give the chunk back its old count.
        expert_idx, old_entry = self.history.check(doc_id)
        if expert_idx >= 0:
            self.hedge.update(expert_idx, ce_score, old_entry.evicted_time, self.time)
            freq = old_entry.freq + 1
        else:
            freq = 1

        if len(self.lru) >= self.capacity:
            self._evict_one()

        self._add(doc_id, _normalize(emb), freq=freq)

    def update_retrieved(self, embeddings: list):
        """Tell the cache what the cold search just returned. RDGE uses this."""
        for emb in embeddings:
            if np.linalg.norm(emb) > 1e-10:
                self._ret_window.append(_normalize(emb))
        if len(self._ret_window) > self.window_size:
            self._ret_window = self._ret_window[-self.window_size:]

    def pop_evictions(self) -> list:
        """Returns the doc_ids removed since the last call."""
        evictions = self._recent_evictions
        self._recent_evictions = []
        return evictions

    def get_stats(self) -> dict:
        total = self.hits + self.misses
        weights = self.hedge.get_weights()
        return {
            "size": self.embed_index.size,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "hits": self.hits,
            "misses": self.misses,
            "weights": {name: round(float(weights[i]), 4) for i, name in enumerate(EXPERT_NAMES)},
            "expert_evictions": {name: int(self._expert_evictions[i]) for i, name in enumerate(EXPERT_NAMES)},
            "weight_updates": self.hedge.n_updates,
            "history_hits": self.history.total_hits,
        }
