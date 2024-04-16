import numpy as np
from utils import stringify_array, all_binary
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


# COSTURIRE l alberi e quindi ogni vettore sarà mappato ad l buckets
# I candidati sono presi dagli l bucket piu vicini
# Una idea potrebbe essere quella di prendere tutti i candidati del bucket piu vicino(chiedere se va bene e ordinarli su questa base)
# APPROCCIO ATTUALE:Prendere dai bucket fin quando non raggiungo k candidati(credo sia quello del Colab)
# APPROCCIO MIGLIORE FORSE: Prendere prima tutti i candidati dei bucket immediatamente piu vicini e se raggiunto k elementi ritornarli.Senno controllare nel secondo livello di vicinanza
# I candidati sono quelli presenti in almeno uno degli l buckets piu vicini
# AGGIUSTARE LA QUESTIONE DI AVERE IL VETTORE STESSO TRA I VICINI

class RandomProjections():
    def __init__(self, d, nbits, l=1, seed=42):
        """
        :param d: Dimensionality of our original vectors(e.g number of users in the dataset)
        :param nbits: Number of hyperplanes
        :param l:Number of threes in the forest
        :self.all_hashes : Given nbits compute all the possible bucket-ids we could have
        :param seed:
        """
        self.nbits = nbits
        self.d = d
        self.l = l
        self.projection_matrix = self._initialize_projection_matrix()
        self.seed = seed
        self.all_hashes = np.vstack(all_binary(nbits))
        if self.seed is not None:
            np.random.seed(self.seed)
        self.buckets_matrix = None

    def candidates_matrix(self, k=5):
        """
        For each vector returns the list of its k candidates
        :return: matrix containing the candidates for each vector in the index
        """
        for index, row in enumerate(self.buckets_matrix.squeeze()):
            candidates = self.search_v2(row, index, k)
            if index == 0:
                candidates_matrix = candidates
            else:
                candidates_matrix = np.vstack([candidates_matrix, candidates])
        return candidates_matrix

    def output_similarities(self, k=5):
        """
        Designed to test LSH using Elliot in a simple way
        :return:
        """
        data, rows_indices, cols_indptr = [], [], []
        candidates = self.candidates_matrix(k)
        for i in range(candidates.shape[0]):
            cols_indptr.append(len(data))
            vec = self._input_matrix[i, :].reshape(1, -1)
            vecs_ = self._input_matrix[candidates[i, :]]
            similarities = cosine_similarity(vec, vecs_)
            data.extend(*similarities)
            rows_indices.extend(*candidates[i, :].reshape(1, -1))
        cols_indptr.append(len(data))
        return data, rows_indices, cols_indptr

    def add(self, input_matrix):
        """
        Inserts each vector into the corresponding bucket(s)
        We have to consider that we could have also a forest of hyperplanes and so each vector will be inserted in l buckets
        1)  self.buckets_matrix : Matrix having as row the bucket relating to the corresponding vector
        2)  self.mapping: an object containing the mappings between buckets and contained vectors
        3)  self.uniques_matrix : Contains only the ids of buckets with content and not all the possible 2^nbits buckets id
        :param input_matrix: Matrix having as rows the vectors we want to bucketize
        """
        # centered_matrix = self.center_ratings_matrix(input_matrix)
        centered_matrix = input_matrix
        self._input_matrix = centered_matrix
        buckets = self.project_matrix(centered_matrix)
        self.buckets_matrix = (buckets > 0).astype(int)
        self.mapping_ = self.create_mappings()

    def create_mappings(self):
        """
        This function contains the logic to map each vector to the l buckets
        :return:
        """
        hash_tables = [defaultdict(list) for _ in range(self.l)]
        for item_idx, buckets in enumerate(self.buckets_matrix.squeeze()):
            for hash_table_id, bucket in enumerate(buckets):
                strigified_id = stringify_array(bucket)
                current_hash_table = hash_tables[hash_table_id]
                current_hash_table[strigified_id].append(item_idx)
        return hash_tables

    def project_matrix(self, input_matrix):
        """
        Take the input matrix and maps each vector in it to k buckets
        :param input_matrix:
        :return:
        """
        local_matrix = input_matrix.copy()
        items, users = input_matrix.toarray().shape
        local_matrix = local_matrix.toarray().reshape(1, items, users)
        return local_matrix.dot(self.projection_matrix)

    def search_v2(self, hashed_vec, index, k=5):
        """
        My approach that is in my opinion more efficient in term of reduction of false positives.
        1) For each vector i pick all the element in the closest buckets in all the l tables
        2) If i have not reached k elements yet i repeat the first step another time checking for the elements in the second's closest buckets
        3) Once reached at lest k elements i compute the similarity returning the element with the highest similarity between the candidates
        TODO: Optimize the code if necessary
        TODO: Remove the index as parameter
        :return:
        """
        candidates = set()
        i = 0
        while True:
            for table_id, el in enumerate(hashed_vec):
                hamming_dist = self.hamming(el, self.all_hashes)
                stringified_id = stringify_array(hamming_dist[i, :])
                candidates = candidates | set(self.mapping_[table_id][stringified_id])
            if i == 0:
                candidates.remove(index)
            if len(candidates) >= k:
                break
            else:
                i = i + 1
        query_vec = self._input_matrix.toarray()[index, :]
        candidates_vectors = self._input_matrix.toarray()[list(candidates)]
        cosine_similarities = cosine_similarity(query_vec.reshape(1, -1), candidates_vectors)
        ordered_indices = np.argsort(cosine_similarities)
        ordered_candidates = np.array(list(candidates))[ordered_indices].squeeze()
        return ordered_candidates[-k:]

    def search(self, hashed_vec, index, k=5):
        """
        First version.
        Source:https://github.com/pinecone-io/examples/blob/master/learn/search/faiss-ebook/locality-sensitive-hashing-random-projection/random_projection.ipynb
        1) I keep taking from the buckets until i reach k elements
        2) I return the first k without caring of the actual similarity between the query and the candidates
        For sure this approach is more efficient but in my opinion could decrease too much the performances
        :return:
        """
        candidates = set()
        i = 0
        while True:
            for table_id, el in enumerate(hashed_vec):
                hamming_dist = self.hamming(el, self.all_hashes)
                stringified_id = stringify_array(hamming_dist[i, :])
                candidates = candidates | set(self.mapping_[table_id][stringified_id])
            candidates.remove(index)
            if len(candidates) >= k:
                break
            else:
                i = i + 1
        return list(candidates)[:k]

    def center_ratings_matrix(self, items_matrix):
        """
        NOT NECESSARY PROBABLY !!!!!
        If i apply RandomProjections with Explicit ratings all the vectors will be on the positive Side of the hyperPlane
        My Solution: Remove the mean for each row!!!!
        :return:
        """
        means = np.mean(items_matrix, axis=1).reshape(-1, 1)
        centered_matrix = items_matrix - means
        return centered_matrix

    def _initialize_projection_matrix(self):
        """
        Useful to inizialize a matrix for projecting our dense vectors in binary ones
        :return:
        """
        return np.random.rand(self.l, self.d, self.nbits) - .5

    def hamming(self, hashed_vec: np.array, other_hashes: np.array) -> np.array:
        """
        Returns the matrix of buckets ordered relatively to the hamming distance
        :param hashed_vec: The bucket assigned to the vector we are considering
        :param other_hashes: All the buckets that have something in it
        :return: Matrix identical to "other_hashes" but ordered relatively to the hamming distance from the current hashed_vec
        """
        # get hamming distance between query vec and all buckets in other_hashes
        hamming_dist = np.count_nonzero(hashed_vec != other_hashes, axis=1).reshape(-1, 1)
        # add hash values to each row
        hamming_dist = np.concatenate((other_hashes, hamming_dist), axis=1)
        # sort based on distance
        hamming_dist = hamming_dist[hamming_dist[:, -1].argsort()][:, :-1]
        return hamming_dist
