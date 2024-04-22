import numpy as np
from utils import stringify_array, all_binary
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import time
from scipy.sparse import csr_matrix
import sparse


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

    def candidates_matrix(self, k=5):
        """
        For each vector returns the list of its k candidates in matrix form
        :return: matrix containing the candidates for each vector in the index
        This approach works wiht self.search function that returns a fixed number of candidates for each vector
        N.B: This approach used in self.output_similarities is faster but force us to select k neighbours in advance
        """
        num_items = self.buckets_matrix.shape[0]
        candidates_matrix = np.empty((num_items, k)).astype(int)
        for index, row in enumerate(self.buckets_matrix):
            candidates = self.search(row, index, k)
            candidates_matrix[index, :] = candidates
        return candidates_matrix

    def output_similarities(self, k=5):
        """
        FASTEST IMPLEMENTATION(even if still slower than pure cosine_similarity itemXitem)
        Calculates the candidate_matrix (contains candidate indices) and from the input matrix (items) selects vectors and calculates their similarity
        :return:
        """
        before = time.time()
        candidates = self.candidates_matrix(k)
        after = time.time()
        print(after - before, "Candidates Matrix")

        index = self._input_matrix.toarray()

        candidates_vectors = index[candidates]

        # Get number of items
        num_items = candidates.shape[0]

        # Initialize the matrix to hold cosine similarities
        similarity_matrix = np.empty((num_items, k))
        # Iniziamo con un for
        prima = time.time()
        for i, vector in enumerate(self._input_matrix):
            prima = time.time()
            similarity_matrix[i, :] = cosine_similarity(vector, candidates_vectors[i])
            dopo = time.time()
            print(dopo - prima, "Single Operation output_similarities")
        dopo = time.time()
        print(dopo - prima, "For loop output_similarities")

        return similarity_matrix, candidates

    def output_similarities_2(self, k=5):
        """
        Slower than self.output_similarities(not too much now) but more flexible in my opinion more correct
        Here instead of having a fixed number of candidates we pick all the candidates(see self.candidates_mapping(k)) and
        starting from the candidates we select the k closest in terms of cosine similarities
        1) Here we have a variable number of candidates for each item
        2) Instead of outputting the first k candidates like in self.candidates_matrix(self.output_similarities) i pick the ones with highest cosine similarity
        :return:
        """
        prima = time.time()
        candidates = self.candidates_mapping(k)
        index = self._input_matrix.toarray()
        candidates_vector = [index[v] for k, v in candidates.items()]
        dopo = time.time()
        print(dopo - prima, "Candidates Mapping")

        # Get number of items
        data, row_indices = [], []
        prima = time.time()
        for i, vector in enumerate(self._input_matrix):
            before = time.time()
            c_similarities = cosine_similarity(vector, candidates_vector[i]).flatten()
            after = time.time()
            print(after - before, "Single Operation outputsimilarities_2")
            sorted_indices = np.argsort(-c_similarities)[:k].flatten()
            data.extend(c_similarities[sorted_indices])
            row_indices.extend(sorted_indices)
        dopo = time.time()
        print(dopo - prima, "For loop output_similarities_2")

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
        Return a python dictionary that has for each item in the input matrix his candidates
        N.B: self.search_1 is different from self.search
        For each vector returns the list of its candidates
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
        """
        This function compute for each buckets it's closest buckets
        This avoid in the search function to compute the hamming distance in each iteration
        :return:
        """
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
        For each bucket, it saves the list of elements that fell into it
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
        Project vectors in the hamming space
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

    def search(self, hashed_vec, index, k=5):
        """
        First version.
        Called in self.candidates_matrix()
        Source:https://github.com/pinecone-io/examples/blob/master/learn/search/faiss-ebook/locality-sensitive-hashing-random-projection/random_projection.ipynb
        1) I keep taking from the buckets until i reach k elements
        2) N.B I return the first k without caring of the actual similarity between the query and the candidates
        !!!For sure this approach is more efficient but we could decrease too much the performance
        N.B: In my opinion we introduce a bias towards to first hash tables picking the first k elements without checking their similarities
        :return:
        """
        candidates = set()
        i = 0
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

    def search_1(self, hashed_vec, index, k=5):
        """
        :param hashed_vec : Vector in binary form
        :param index : Index of the vector in the input_matrix
        :param k: Number of neighbours (parameter k of Item-KNN)
        Called by
        This version has the flexibility of returning more than k candidates without picking just the first k candidates
        1) I keep taking from the buckets until i reach k elements
         1a) N.B Each time i iterate on the closest buckets(in term of hamming) of each hash table.
         1b) If i have not yet k candidates i move to the second closest bucket and so on
        2) Return all the candidates
        N.B This function does not work in self.candidates_matrix because it returns a different number /
        of candidates for each vector (at least 5) and thus makes it impossible to create a matrix
        :return:
        """
        candidates = set()
        i = 0
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

    def _initialize_projection_matrix(self):
        """
        Useful to inizialize a matrix for projecting our dense vectors in binary ones
        :return:
        """
        return np.random.rand(self.l, self.d, self.nbits) - .5
