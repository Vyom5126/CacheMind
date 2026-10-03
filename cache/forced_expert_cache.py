"""
A cache that always follows one expert and never learns. The benchmark
uses it to compare SemLRU, LFU and RDGE on their own against the learned
cache, with everything else kept the same.
"""

from cache.experts import N_EXPERTS, get_all_nominees
from cache.scrl_cache import SCRLCache


class ForcedExpertCache(SCRLCache):

    def __init__(self, *args, forced_expert: int, **kwargs):
        super().__init__(*args, **kwargs)
        assert 0 <= forced_expert < N_EXPERTS
        self.forced_expert = forced_expert

    def _evict_one(self):
        nominees = get_all_nominees(self.embed_index, self.lfu, self._q_window, self._ret_window)

        evict_id = nominees[self.forced_expert]
        if evict_id is None:
            evict_id = next((n for n in nominees if n is not None), None)
        if evict_id is None:
            return

        entry = self.lru[evict_id]
        entry.evicted_time = self.time
        self._remove(evict_id)
        self._recent_evictions.append(evict_id)
        self._expert_evictions[self.forced_expert] += 1
