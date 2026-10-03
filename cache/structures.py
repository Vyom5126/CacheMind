"""
Two small data structures, adapted from the LeCaR code.

DequeDict: a dict that remembers order. The oldest item is at the front,
           and adding or removing an item is O(1). Used for recency.
HeapDict:  a dict whose smallest item can be read in O(1). Used for LFU.
"""


class DequeDict:

    class Node:
        __slots__ = ["key", "value", "prev", "next"]

        def __init__(self, key, value):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None

    def __init__(self):
        self.htbl = {}
        self.head = None   # oldest
        self.tail = None   # newest

    def __contains__(self, key):
        return key in self.htbl

    def __len__(self):
        return len(self.htbl)

    def __getitem__(self, key):
        return self.htbl[key].value

    def __setitem__(self, key, value):
        # Setting a key again moves it to the newest end.
        if key in self.htbl:
            self._remove(key)
        self._push(key, value)

    def __delitem__(self, key):
        self._remove(key)

    def first(self):
        return self.head.value

    def popFirst(self):
        node = self.head
        self._remove(node.key)
        return node.value

    def _push(self, key, value):
        node = self.Node(key, value)
        self.htbl[key] = node
        if self.tail:
            self.tail.next = node
            node.prev = self.tail
        else:
            self.head = node
        self.tail = node

    def _remove(self, key):
        node = self.htbl.pop(key)
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev


class HeapDict:

    class Node:
        __slots__ = ["key", "value", "index"]

        def __init__(self, key, value):
            self.key = key
            self.value = value
            self.index = -1

        def __lt__(self, other):
            return self.value < other.value

    def __init__(self):
        self.htbl = {}
        self.heap = []

    def __contains__(self, key):
        return key in self.htbl

    def __len__(self):
        return len(self.heap)

    def __getitem__(self, key):
        return self.htbl[key].value

    def min(self):
        return self.heap[0].value if self.heap else None

    def popMin(self):
        node = self.heap[0]
        del self[node.key]
        return node.value

    def __setitem__(self, key, value):
        if key in self.htbl:
            # The value may have changed, so move it up or down to its new place.
            node = self.htbl[key]
            node.value = value
            self._up(node.index)
            self._down(node.index)
        else:
            node = self.Node(key, value)
            self.htbl[key] = node
            node.index = len(self.heap)
            self.heap.append(node)
            self._up(node.index)

    def __delitem__(self, key):
        node = self.htbl.pop(key)
        last = self.heap[-1]
        self._swap(node, last)
        self.heap.pop()
        if node is not last:
            self._up(last.index)
            self._down(last.index)

    def _swap(self, a, b):
        a.index, b.index = b.index, a.index
        self.heap[a.index] = a
        self.heap[b.index] = b

    def _up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self.heap[i] < self.heap[parent]:
                self._swap(self.heap[i], self.heap[parent])
                i = parent
            else:
                break

    def _down(self, i):
        n = len(self.heap)
        while True:
            smallest = i
            left, right = 2 * i + 1, 2 * i + 2
            if left < n and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < n and self.heap[right] < self.heap[smallest]:
                smallest = right
            if smallest == i:
                break
            self._swap(self.heap[i], self.heap[smallest])
            i = smallest
