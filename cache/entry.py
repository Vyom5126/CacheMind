class SCRLEntry:
    """One cached chunk. The same object is stored in both the LRU list and
    the LFU heap, so updating it once updates it everywhere."""

    __slots__ = ["doc_id", "freq", "time", "evicted_time"]

    def __init__(self, doc_id: str, freq: int = 1, time: int = 0):
        self.doc_id = doc_id
        self.freq = freq            # how many times it has been used
        self.time = time            # question number when it was last used
        self.evicted_time = None    # question number when it was removed

    def __lt__(self, other):
        # Used by the LFU heap: fewer uses goes first, and on a tie the
        # one that was used longer ago goes first.
        if self.freq == other.freq:
            return self.time < other.time
        return self.freq < other.freq

    def __repr__(self):
        return f"SCRLEntry(id={self.doc_id}, freq={self.freq}, time={self.time}, evicted={self.evicted_time})"
