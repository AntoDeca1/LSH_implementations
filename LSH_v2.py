import numpy as np
from utils import stringify_array, all_binary
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import time


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
        num_items = self.buckets_matrix.shape[0]
        candidates_matrix = np.zeros((num_items, k)).astype(int)
        for index, row in enumerate(self.buckets_matrix):
            candidates = self.search(row, index, k)
            candidates_matrix[index, :] = candidates
        return candidates_matrix

    def output_similarities(self, k=5):
        """
        Designed to test LSH using Elliot in a simple way
        QUA POSSO PROVARE CON IL MAPPING SENZA LA CANDIDATES MATRIX
        Se qui uso un mapping con un vettore associato  posso risolvere forse in maniera intelligente
        Devo solo bloccare dopo i primi k elementi
        Provare a migliorare questo in termini di efficienza con una unica operazione matriciale
        :return:
        """
        prima = time.time()
        candidates = self.candidates_matrix(k)
        dopo = time.time()
        print(dopo - prima, "Candidates Matrix")

        index = self._input_matrix.toarray()

        candidates_vectors = index[candidates]

        # Get number of items
        num_items = candidates.shape[0]

        # Initialize the matrix to hold cosine similarities
        similarity_matrix = np.empty((num_items, k))
        # Iniziamo con un for

        for i, vector in enumerate(self._input_matrix):
            prima = time.time()
            similarity_matrix[i, :] = cosine_similarity(vector, candidates_vectors[i])
            dopo = time.time()
            print(dopo - prima, "For loop output_similarities")

        return similarity_matrix, candidates

    def output_similarities_3(self, k=5):
        prima = time.time()
        candidates = self.candidates_list(k)  # This should return a list of candidate indices for each item
        dopo = time.time()
        print(dopo - prima, "Candidates Mapping")

        # Flatten the list of candidate indices to fetch all vectors at once
        all_candidate_indices = np.concatenate(candidates)
        all_candidate_vectors = self._input_matrix[all_candidate_indices]

        # Now create an array to map back each candidate to its original item
        item_index = np.repeat(np.arange(len(candidates)), [len(c) for c in candidates])

        # Compute cosine similarities in batch
        prima = time.time()
        all_similarities = cosine_similarity(self._input_matrix, all_candidate_vectors)
        dopo = time.time()
        print(dopo - prima, "Cosine similarity batch computation")

        # Sort and select the top k for each item
        similarity_matrix = []
        start_index = 0
        for i, size in enumerate([len(c) for c in candidates]):
            end_index = start_index + size
            item_similarities = all_similarities[i, start_index:end_index]
            sorted_indices = np.argsort(-item_similarities)[:k]
            top_k_similarities = item_similarities[sorted_indices]
            similarity_matrix.append(top_k_similarities)
            start_index = end_index

        similarity_matrix = np.array(similarity_matrix)

        return similarity_matrix, candidates

    def output_similarities_2(self, k=5):
        """
        Designed to test LSH using Elliot in a simple way
        QUA POSSO PROVARE CON IL MAPPING SENZA LA CANDIDATES MATRIX
        Se qui uso un mapping con un vettore associato  posso risolvere forse in maniera intelligente
        Devo solo bloccare dopo i primi k elementi
        Provare a migliorare questo in termini di efficienza con una unica operazione matriciale
        :return:
        """
        prima = time.time()
        candidates = self.candidates_mapping(k)
        dopo = time.time()
        print(dopo - prima, "Candidates Mapping")

        # Get number of items
        data, row_indices = [], []

        for i, vector in enumerate(self._input_matrix):
            prima = time.time()
            candidates_vectors = self._input_matrix[candidates[i]]
            dopo = time.time()
            print(dopo - prima, "For loop output_similarities_2")
            c_similarities = cosine_similarity(vector, candidates_vectors).flatten()
            # sorted_indices = np.argsort(-c_similarities)[:k].flatten()
            # data.extend(c_similarities[sorted_indices])
            # row_indices.extend(sorted_indices)

        return data, row_indices

    def candidates_list(self, k):
        """
                For each vector returns the list of its k candidates
                :return: matrix containing the candidates for each vector in the index
                """
        candidates_mapping = []
        for index, row in enumerate(self.buckets_matrix):
            candidates = self.search_1(row, index, k)
            candidates_mapping.append(candidates)

        return candidates_mapping

    def candidates_mapping(self, k):
        """
                For each vector returns the list of its k candidates
                :return: matrix containing the candidates for each vector in the index
                """
        candidates_mapping = {}
        for index, row in enumerate(self.buckets_matrix):
            candidates = self.search_1(row, index, k)
            candidates_mapping[index] = candidates

        return candidates_mapping

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
        centered_matrix = input_matrix
        self._input_matrix = centered_matrix
        buckets = self.project_matrix(centered_matrix)
        self.buckets_matrix = (buckets > 0).astype(int)
        self.all_hashes = np.unique(self.buckets_matrix.reshape(-1, self.nbits), axis=0)
        self.mapping_ = self.create_mappings()
        self.closest_buckets = self.precompute_closest_buckets()  # Da verificare se ne vale la pena precalcolarle

    def precompute_closest_buckets(self):
        # Dictionary to store the closest buckets for each hash
        closest_buckets = {}
        num_hashes = self.all_hashes.shape[0]

        # Compute the Hamming distance between each pair of hashes
        expanded_hashes = np.expand_dims(self.all_hashes, 1)
        distances = np.count_nonzero(expanded_hashes != expanded_hashes.transpose(1, 0, 2), axis=2)

        # For each hash, find and store indices of the closest buckets
        for idx in range(num_hashes):
            sorted_indices = np.argsort(distances[idx])
            # Save the closest buckets' indices, skip the first one as it is the hash itself
            closest_buckets[stringify_array(self.all_hashes[idx])] = sorted_indices  # Adjust slicing as needed

        return closest_buckets

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

    def search_1(self, hashed_vec, index, k=5):
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
        # Expand A to compare against every row of B

        while True:
            for table_id, el in enumerate(hashed_vec):
                id = stringify_array(self.all_hashes[self.closest_buckets[stringify_array(el)][i]])
                candidates = candidates | set(self.mapping_[table_id][id])
            if i == 0:
                candidates.remove(index)
            if len(candidates) >= k:
                break
            else:
                i = i + 1
        return list(candidates)

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
        # Expand A to compare against every row of B

        while True:
            for table_id, el in enumerate(hashed_vec):
                id = stringify_array(self.all_hashes[self.closest_buckets[stringify_array(el)][i]])
                candidates = candidates | set(self.mapping_[table_id][id])
            if i == 0:
                candidates.remove(index)
            if len(candidates) >= k:
                break
            else:
                i = i + 1
        return list(candidates)[:k]

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
