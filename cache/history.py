"""
The ghost cache: one short list per expert of the chunks it removed
recently. Only the record is kept, not the embedding, so it takes no room
in the real cache.

If a removed chunk comes back from the cold search and is still in an
expert's list, that removal was a mistake and the expert gets penalized.
"""

from cache.entry import SCRLEntry
from cache.experts import N_EXPERTS
from cache.structures import DequeDict


class HistoryQueues:

    def __init__(self, hist_size: int):
        self.hist_size = hist_size
        self.queues = [DequeDict() for _ in range(N_EXPERTS)]
        self.total_hits = 0   # how many times a removed chunk was needed again

    def add(self, entry: SCRLEntry, expert_idx: int):
        # -1 means all experts picked the same chunk, so nobody is to blame.
        if expert_idx < 0:
            return

        queue = self.queues[expert_idx]
        if len(queue) == self.hist_size:
            queue.popFirst()
        queue[entry.doc_id] = entry

    def check(self, doc_id: str) -> tuple:
        """If doc_id is in an expert's list, remove it and return
        (expert_idx, entry). Otherwise return (-1, None)."""
        for i, queue in enumerate(self.queues):
            if doc_id in queue:
                entry = queue[doc_id]
                del queue[doc_id]
                self.total_hits += 1
                return i, entry
        return -1, None

    def sizes(self) -> list:
        return [len(q) for q in self.queues]
