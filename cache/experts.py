"""
The three eviction experts. When the cache is full, each one picks a chunk
to remove, and the cache decides whose pick to follow.

SemLRU: removes the chunk least similar to recent questions.
LFU:    removes the chunk used the fewest times.
RDGE:   like SemLRU, but also keeps chunks similar to what the cold
        search returned recently.
"""

from typing import Optional

import numpy as np

from cache.embed_index import EmbedIndex
from cache.structures import HeapDict

SEM_LRU, LFU, RDGE = 0, 1, 2
N_EXPERTS = 3
EXPERT_NAMES = ["SemLRU", "LFU", "RDGE"]


def pick_lfu(lfu: HeapDict) -> Optional[str]:
    entry = lfu.min()
    return entry.doc_id if entry else None


def pick_semantic_lru(index: EmbedIndex, recent_queries: list, lfu: HeapDict) -> Optional[str]:
    if not recent_queries:
        return pick_lfu(lfu)

    doc_ids, doc_matrix = index.get_doc_matrix()
    if doc_matrix is None:
        return None

    # A chunk is worth keeping if it is close to at least one recent question.
    queries = np.array(recent_queries, dtype=np.float32)
    keep_score = (doc_matrix @ queries.T).max(axis=1)
    return doc_ids[int(np.argmin(keep_score))]


def pick_rdge(index: EmbedIndex, recent_queries: list, recent_retrieved: list,
              lfu: HeapDict) -> Optional[str]:
    if not recent_queries and not recent_retrieved:
        return pick_lfu(lfu)

    doc_ids, doc_matrix = index.get_doc_matrix()
    if doc_matrix is None:
        return None

    # Same idea as SemLRU, but being close to a recent cold-search result
    # also counts as a reason to keep a chunk.
    keep_score = np.zeros(len(doc_ids), dtype=np.float32)
    if recent_queries:
        queries = np.array(recent_queries, dtype=np.float32)
        keep_score = np.maximum(keep_score, (doc_matrix @ queries.T).max(axis=1))
    if recent_retrieved:
        retrieved = np.array(recent_retrieved, dtype=np.float32)
        keep_score = np.maximum(keep_score, (doc_matrix @ retrieved.T).max(axis=1))

    return doc_ids[int(np.argmin(keep_score))]


def get_all_nominees(index: EmbedIndex, lfu: HeapDict, recent_queries: list,
                     recent_retrieved: list) -> list:
    """Returns one pick per expert, in the order SemLRU, LFU, RDGE."""
    return [
        pick_semantic_lru(index, recent_queries, lfu),
        pick_lfu(lfu),
        pick_rdge(index, recent_queries, recent_retrieved, lfu),
    ]
