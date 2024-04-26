import faiss
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class LSH_faiss:
    def __init__(self, d, l, nbits):
        self.d = d
        self.l = l
        self.nbits = nbits
        self.indices = [faiss.IndexLSH(d, nbits) for _ in range(self.l)]

    def _initialize_indices(self):
        indices = []
        for i in range(self.l):
            index = faiss.IndexLSH(self.d, self.nbits)
            indices.append(index)

    def add(self, input):
        self._input = input
        for i, index in enumerate(self.indices):
            index.rrot.init(42 + i + 12)
            index.add(input)

    def search(self, k):
        for row_n, row in enumerate(self._input):
            candidates = set()
            for index in self.indices:
                _, I = index.search(row.reshape(1, -1), k)
                candidates = candidates | set(I[0])
            similarities = cosine_similarity(row.reshape(1, -1), self._input[list(candidates)])
        return candidates
