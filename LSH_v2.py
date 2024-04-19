import numpy as np
from utils import stringify_array, all_binary
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import time


# RIMUOVERE TUTTI QUESTI TRANSPOSE AGENDO SULLA FUNZIONE PROJECT MATRIX

# DA OTTIMIZZARE
# Potrebbe essere da ottimizzare come calcolo la cosine similairity.
# IDEA:Potrei prendere i candidati da tutti i bucket di livello 1 e continuare a prendere tutti i candidati fin quando non ne ho almeno k
# Trovare una maniera di calcolare le similarità in maniera matriciale magari sfruttando le matrici di scipy che sembrano velocizzare il calcolo

class RandomProjections():
    def __init__(self, d, nbits, l=1, seed=42):
        """
        :param d: Dimensionality of our original vectors(e.g number of users in the dataset)
        :param nbits: Number of hyperplanes
        :param l: Number of threes in the forest
        :self.all_hashes : Given nbits compute all the possible bucket-ids we could have
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

    def candidates_matrix(self, k=5):
        """
        For each vector returns the list of its k candidates
        :return: matrix containing the candidates for each vector in the index
        """
        for index, row in enumerate(self.buckets_matrix):
            candidates = self.search(row, index, k)
            if index == 0:
                candidates_matrix = candidates
            else:
                candidates_matrix = np.vstack([candidates_matrix, candidates])
        print()
        return candidates_matrix

    def output_similarities(self, k=5):
        """
        Designed to test LSH using Elliot in a simple way
        :return:
        """
        candidates = self.candidates_matrix(k)

        index = self._input_matrix.toarray()

        candidates_vectors = index[candidates]

        # Get number of items
        num_items = candidates.shape[0]

        # Initialize the matrix to hold cosine similarities
        similarity_matrix = np.zeros((num_items, k))
        # Iniziamo con un for

        for i, vector in enumerate(self._input_matrix):
            similarity_matrix[i, :] = cosine_similarity(vector, candidates_vectors[i])

        return similarity_matrix

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
        self.all_hashes = np.unique(self.buckets_matrix.reshape(-1, self.nbits), axis=0)
        self.mapping_ = self.create_mappings()
        print()

    def create_mappings(self):
        """
        This function contains the logic to map each vector to the l buckets
        :return:
        """
        hash_tables = [defaultdict(list) for _ in range(self.l)]
        for item_idx, buckets in enumerate(self.buckets_matrix):
            for hash_table_id, bucket in enumerate(buckets):
                strigified_id = stringify_array(bucket)
                current_hash_table = hash_tables[hash_table_id]
                current_hash_table[strigified_id].append(item_idx)
        print()
        return hash_tables

    def project_matrix(self, input_matrix):
        """
        Take the input matrix and maps each vector in it to k buckets
        :param input_matrix:
        :return:
        """
        output = None
        for i in range(self.projection_matrix.shape[0]):
            temp = input_matrix.dot(self.projection_matrix[i])
            temp = temp.reshape(1, *temp.shape)
            if output is None:
                output = temp
            else:
                output = np.concatenate([output, temp])
        return output.transpose(1, 0, 2)

    def search_v2(self, hashed_vec, index, k=5):
        """
        My approach that is in my opinion more efficient in term of reduction of false positives.
        1) For each vector i pick all the element in the closest buckets in all the l tables
        2) If i have not reached k elements yet i repeat the first step another time checking for the elements in the second's closest buckets
        3) Once reached at least k elements i compute the similarity returning the k elements with the highest similarity between the candidates
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
        Supponiamo che search vada bene
        Source:https://github.com/pinecone-io/examples/blob/master/learn/search/faiss-ebook/locality-sensitive-hashing-random-projection/random_projection.ipynb
        1) I keep taking from the buckets until i reach k elements
        2) I return the first k without caring of the actual similarity between the query and the candidates
        For sure this approach is more efficient but in my opinion could decrease too much the performance
        Optimize the removal of indexes
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
