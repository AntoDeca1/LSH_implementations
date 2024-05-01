import faiss
import numpy as np
import scipy.sparse as sp


class LSH:
    def __init__(self, d, nbits):
        """
        :param d: Dimensionality of the original vectors
        :param nbits: Dimensionality of the projected vectors(e.g number of hyperplanes)
        """
        self.d = d
        self.nbits = nbits
        self.index = faiss.IndexLSH(d, nbits)

    def add(self, input_matrix):
        self._input_matrix = input_matrix
        # Controllo sul tipo della input matrix input_matrix
        self.index.add(input_matrix)

    def search(self, input_matrix, k=20):
        n = len(input_matrix)
        output_matrix = np.zeros((n, n), dtype=int)
        D, I = self.index.search(input_matrix, k)
        for index, row in enumerate(I):
            output_matrix[index, row] = 1
        return sp.csr_matrix(output_matrix)

    def search_2(self, input_matrix, k=20):
        _, I = self.index.search(input_matrix, k)
        return I
