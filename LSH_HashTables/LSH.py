from table import HashTable
import numpy as np
import scipy.sparse as sp


class LSHRp:
    def __init__(self, dim, num_tables, hash_size):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.tables = [HashTable(self.hash_size, dim) for _ in range(self.num_tables)]

    def add(self, vecs):
        """
        Bucket the vector in each table
        :param vecs:
        :return:
        """
        for table in self.tables:
            table.add(vecs)

    def query(self, vecs):
        n = vecs.shape[0]
        output_matrix = np.zeros((n, n), dtype=int)
        for table in self.tables:
            candidates = table.query(vecs)
            for index, candidate in enumerate(candidates):
                output_matrix[index, candidate] = 1
        # RITORNARE DIRETTAMENTE LA MATRICE DI NUMPY rende piu rapido il calcolo
        # return output_matrix
        return sp.csr_matrix(output_matrix)
