import sys
import matplotlib.pyplot as plt
import numpy as np
from utils import stringify_array
from collections import defaultdict
import scipy.sparse as sp

"""
In questa implementazione
1) Seguo lo scheletro iniziale(LSH_v2)
2) Lo adatto all'approccio https://musicinformationretrieval.com/lsh_fingerprinting.html

Prendo come candidati solo quelli che ricadono nello stesso bucket per almeno una hash table.(Non uso hamming distance)
N.B:Può succedere che non ci siano candidati in questo caso
"""


class RandomProjections():
    def __init__(self, d, nbits, l=1, seed=42):
        """
        :param d: Dimensionality of our original vectors (e.g number of users in the dataset)
        :param nbits: Number of hyperplanes
        :param l: Number of threes in the forest
        :self.all_hashes : The hashes of the buckets that contain something in
        :param seed:
        """
        self.nbits = nbits
        self.d = d
        self.l = l
        self.projection_matrix = self._initialize_projection_matrix()
        self.seed = seed
        self.all_hashes = None
        if self.seed is not None:
            np.random.seed(self.seed)
        self.buckets_matrix = None

    def add(self, input_matrix):
        """
        Inserts each vector into the corresponding bucket(s)
        We have to consider that we could have also a forest of hyperplanes and so each vector will be inserted in l buckets
        1)  self.buckets_matrix : Matrix having as row the bucket relating to the corresponding vector
        2)  self.mapping: an object containing the mappings between buckets and contained vectors
        3)  self.all_hashes : Contains only the ids of buckets with content and not all the possible 2^nbits buckets id
        4)  self.closest_buckets: Pre-Computed closest buckets for each bucket in term of hamming distance(SLOW THE INDEX BUT MAKES THE SEARCH FASTER)
        :param input_matrix: Matrix having as rows the vectors we want to bucketize
        """
        buckets = self.project_matrix(input_matrix)
        self.buckets_matrix = (buckets > 0).astype(int)
        self.mapping_ = self.create_mappings()

    def project_matrix(self, input_matrix):
        """
        Project vectors in the hamming space
        :param input_matrix:
        :return:
        """
        output = np.empty((self.l, input_matrix.shape[0], self.nbits))
        for i in range(self.projection_matrix.shape[0]):
            temp = input_matrix.dot(self.projection_matrix[i])
            temp = np.expand_dims(temp, axis=0)
            output[i] = temp
        return output.transpose(1, 0, 2)

    def _get_vec_candidates(self, vec):
        """
        Given a vector rapresentation(already bucketized) pick it's candidates
        :param vec: shape(n_tables,nbits)
        :return:
        """

        candidates = set()
        for index, table in enumerate(self.mapping_):
            candidates.update(table[stringify_array(vec[index])])
        return candidates

    def search_2(self):
        """
        For each element pick it's candidates
        :return: A sparse matrix of dimensionality (n_items,n_items) or (n_users,n_users) having 1 only in the candidates position
        """
        n = len(self.buckets_matrix)
        # data, indices, cols_ranges = [], [], []
        output_matrix = np.zeros((n, n), dtype=int)
        for index, el in enumerate(self.buckets_matrix):
            candidates = list(self._get_vec_candidates(el))
            output_matrix[index, candidates] = 1
        # N.B Non passare alla scipy matrix rende il tutto piu rapido
        # return output_matrix
        return sp.csr_matrix(output_matrix)

    def create_mappings(self):
        """
        For each bucket and for each table, it saves the list of elements that fell into it
        :return:
        """
        hash_tables = [defaultdict(list) for _ in range(self.l)]
        for item_idx, buckets in enumerate(self.buckets_matrix):
            for hash_table_id, bucket in enumerate(buckets):
                strigified_id = stringify_array(bucket)
                current_hash_table = hash_tables[hash_table_id]
                current_hash_table[strigified_id].append(item_idx)
        return hash_tables

    def _initialize_projection_matrix(self):
        """
        Useful to inizialize a matrix for projecting our dense vectors in binary ones
        :return:
        """
        return np.random.rand(self.l, self.d, self.nbits) - .5

    def plot_bucket_occupation(self):
        total_buckets = 2 ** self.nbits
        num_tables = len(self.mapping_)

        filled_buckets_list = []
        avg_elements_per_bucket_list = []

        for i, table in enumerate(self.mapping_):
            filled_buckets = len(table)
            filled_buckets_list.append(filled_buckets)

            # Calculate the average number of elements per filled bucket
            if filled_buckets > 0:
                avg_elements_per_bucket = sum(len(candidates) for candidates in table.values()) / filled_buckets
            else:
                avg_elements_per_bucket = 0
            avg_elements_per_bucket_list.append(avg_elements_per_bucket)

            print(f"Hash Table {i + 1}: {filled_buckets} out of {total_buckets} buckets are filled.")

            # Prepare data for plotting
            bucket_sizes = [len(candidates) for candidates in table.values()]
            bucket_ids = list(table.keys())

            # Plotting the data
            plt.figure(figsize=(12, 6))
            plt.bar(bucket_ids, bucket_sizes, color='skyblue', width=0.6)
            plt.xlabel('Bucket ID', fontsize=12)
            plt.ylabel('Number of Candidates', fontsize=12)
            plt.title(f'Occupation of Buckets in Hash Table {i + 1}', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

        # Plot the number of filled buckets for each hash table
        plt.figure(figsize=(12, 6))
        plt.bar(range(1, num_tables + 1), filled_buckets_list, color='skyblue', width=0.6)
        plt.xlabel('Hash Table', fontsize=12)
        plt.ylabel('Number of Filled Buckets', fontsize=12)
        plt.title('Number of Filled Buckets per Hash Table', fontsize=14)
        plt.xticks(range(1, num_tables + 1))
        plt.tight_layout()
        plt.show()

        # Plot the average number of elements per filled bucket for each hash table
        plt.figure(figsize=(12, 6))
        plt.bar(range(1, num_tables + 1), avg_elements_per_bucket_list, color='skyblue', width=0.6)
        plt.xlabel('Hash Table', fontsize=12)
        plt.ylabel('Average Number of Elements per Filled Bucket', fontsize=12)
        plt.title('Average Number of Elements per Filled Bucket per Hash Table', fontsize=14)
        plt.xticks(range(1, num_tables + 1))
        plt.tight_layout()
        plt.show()
