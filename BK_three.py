import numpy as np


class BKNode:
    def __init__(self, hash_code):
        self.hash = hash_code
        self.children = {}


class BKTree:
    def __init__(self):
        self.root = None

    def insert(self, hash_code):
        if self.root is None:
            self.root = BKNode(hash_code)
        else:
            node = self.root
            while True:
                distance = self.hamming_distance(node.hash, hash_code)
                if distance in node.children:
                    node = node.children[distance]
                else:
                    node.children[distance] = BKNode(hash_code)
                    break

    def query(self, hash_code, max_distance):
        candidates = []
        self._query_recursive(self.root, hash_code, max_distance, candidates)
        return candidates

    def _query_recursive(self, node, hash_code, max_distance, candidates):
        if node is None:
            return
        dist = self.hamming_distance(node.hash, hash_code)
        if dist <= max_distance:
            candidates.append(node.hash)
        for d in range(dist - max_distance, dist + max_distance + 1):
            if d in node.children:
                self._query_recursive(node.children[d], hash_code, max_distance, candidates)

    @staticmethod
    def hamming_distance(h1, h2):
        return np.count_nonzero(h1 != h2)
