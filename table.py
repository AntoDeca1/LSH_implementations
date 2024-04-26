import os
import os.path
import numpy as np
from collections import defaultdict
from utils import stringify_array


class HashTable:
    def __init__(self, hash_size, dim):
        self.table = defaultdict(list)
        self.hash_size = hash_size
        self.projections = np.random.randn(self.hash_size, dim)

    def add(self, input_matrix):
        hashes = self._project(input_matrix)
        for index, hashing in enumerate(hashes):
            self.table[stringify_array(hashing)].append(index)

    def _project(self, input_matrix):
        signatures = input_matrix.dot(self.projections.T)
        return (signatures > 0).astype(int)

    def query(self, vecs):
        """
        Versione non ottimizzata
        :param vecs:
        :return:
        """
        # Candidates ha per ogni vettore in vecs i vicini candidati(quelli che cadono nello stesso bucket)
        candidates = []
        # Un hash per ogni vettore
        hashes = self._project(vecs)
        for hash in hashes:
            candidates.append(self.table[stringify_array(hash)])
        return candidates
