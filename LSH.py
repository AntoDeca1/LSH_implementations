import librosa
from table import HashTable
import numpy as np
from collections import defaultdict
import scipy.sparse as sp


class LSHRp:
    def __init__(self, dim, num_tables, hash_size):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.tables = list()
        for i in range(self.num_tables):
            self.tables.append(HashTable(self.hash_size, dim))

    def add(self, vecs):
        """
        Bucket the vector in each table
        :param vecs:
        :return:
        """
        for table in self.tables:
            table.add(vecs)

    def query(self, vecs):
        full_candidates = defaultdict(set)
        for table in self.tables:
            candidates = table.query(vecs)
            for index, candidate in enumerate(candidates):
                full_candidates[index] = full_candidates[index] | set(candidate)
        return self.construct_datastructure(full_candidates)

    def construct_datastructure(self, full_candidates):
        n = len(full_candidates)
        matrix = sp.dok_matrix((n, n), dtype=int)
        for key, indices in full_candidates.items():
            for index in indices:
                matrix[key, index] = 1
        return matrix.tocsr()

    def describe(self):
        pass
