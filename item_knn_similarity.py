import pickle

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, haversine_distances, chi2_kernel, \
    manhattan_distances
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from LSH_v3.LSH_v3 import RandomProjections
import time
import scipy.sparse as sp


class Similarity(object):
    """
    Simple kNN class
    """

    def __init__(self, data, num_neighbors, similarity, implicit, nbits, ntables):
        self._data = data
        self._ratings = data.train_dict
        self._num_neighbors = num_neighbors
        self._similarity = similarity
        self._implicit = implicit

        # CODICE AGGIUNTO PER LSH
        self._nbits = nbits
        self._ntables = ntables

        if self._implicit:
            self._URM = self._data.sp_i_train
        else:
            self._URM = self._data.sp_i_train_ratings

        self._users = self._data.users
        self._items = self._data.items
        self._private_users = self._data.private_users
        self._public_users = self._data.public_users
        self._private_items = self._data.private_items
        self._public_items = self._data.public_items

    def initialize(self):
        """
        This function initialize the data model
        """

        self.supported_similarities = ["cosine", "dot", ]
        self.supported_dissimilarities = ["euclidean", "manhattan", "haversine", "chi2", 'cityblock', 'l1', 'l2',
                                          'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming',
                                          'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto',
                                          'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
                                          'yule']
        print(f"\nSupported Similarities: {self.supported_similarities}")
        print(f"Supported Distances/Dissimilarities: {self.supported_dissimilarities}\n")

        # NEW CODE ADDED HERE
        data, rows_indices, cols_indptr = [], [], []
        print(f"Similarity used : {self._similarity} with {self._num_neighbors} neighbors")
        self._similarity_matrix = np.zeros((len(self._items), len(self._items)))
        if (self._similarity == "lsh_rp"):
            print(f"We are using nbits: {self._nbits} and ntables: {self._ntables} ")
            rp = RandomProjections(d=len(self._users), nbits=self._nbits, l=self._ntables, seed=42)
            prima = time.time()
            rp.add(self._URM.T)
            dopo = time.time()
            print(dopo - prima, "Tempo per indicizzare la similarity matrix")
            prima = time.time()
            candidates_matrix = rp.output_candidates()
            dopo = time.time()
            print(dopo - prima, "Tempo per tirare fuori i candidati")
            prima = time.time()
            self.compute_candidates_cosine_similarity(self._URM.T, candidates_matrix)
            dopo = time.time()
            print(dopo - prima, "Tempo per calcolare la similarity matrix")
        else:
            self.process_similarity(self._similarity)

        data, rows_indices, cols_indptr = [], [], []

        column_row_index = np.arange(len(self._data.items), dtype=np.int32)

        for item_idx in range(len(self._data.items)):
            cols_indptr.append(len(data))
            column_data = self._similarity_matrix[:, item_idx]

            non_zero_data = column_data != 0
            # N:B: A me in fin dei conti serve restituire similarità e indici nel range 0-numitems dei piu similari
            idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
            top_k_idx = idx_sorted[-self._num_neighbors:]
            data.extend(column_data[non_zero_data][top_k_idx])
            rows_indices.extend(column_row_index[non_zero_data][top_k_idx])

        cols_indptr.append(len(data))

        W_sparse = sparse.csc_matrix((data, rows_indices, cols_indptr),
                                     shape=(len(self._data.items), len(self._data.items)), dtype=np.float32).tocsr()
        self._preds = self._URM.dot(W_sparse).toarray()

        del self._similarity_matrix

    def process_similarity(self, similarity):
        if similarity == "cosine":
            self._similarity_matrix = cosine_similarity(self._URM.T)
        elif similarity == "dot":
            self._similarity_matrix = (self._URM.T @ self._URM).toarray()
        elif similarity == "euclidean":
            self._similarity_matrix = (1 / (1 + euclidean_distances(self._URM.T)))
        elif similarity == "manhattan":
            self._similarity_matrix = (1 / (1 + manhattan_distances(self._URM.T)))
        elif similarity == "haversine":
            self._similarity_matrix = (1 / (1 + haversine_distances(self._URM.T)))
        elif similarity == "chi2":
            self._similarity_matrix = (1 / (1 + chi2_kernel(self._URM.T)))
        elif similarity in ['cityblock', 'l1', 'l2']:
            self._similarity_matrix = (1 / (1 + pairwise_distances(self._URM.T, metric=similarity)))
        elif similarity in ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard',
                            'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                            'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']:

            self._similarity_matrix = (1 / (1 + pairwise_distances(self._URM.T.toarray(), metric=similarity)))
        else:
            raise ValueError("Compute Similarity: value for parameter 'similarity' not recognized."
                             f"\nAllowed values are: {self.supported_similarities}, {self.supported_dissimilarities}."
                             f"\nPassed value was {similarity}\nTry with implementation: aiolli")

    def compute_candidates_cosine_similarity(self, item_user_matrix, candidate_matrix):
        """
        Funzione che richiede tempo
        Esempi:
        We are using nbits: 6 and ntables: 10
        0.14770293235778809 Tempo per indicizzare la similarity matrix
        0.6376771926879883 Tempo per tirare fuori i candidati
        14.769452095031738 Tempo per calcolare la similarity matrix
        We are using nbits: 8 and ntables: 10
        0.16554594039916992 Tempo per indicizzare la similarity matrix
        0.30977892875671387 Tempo per tirare fuori i candidati
        8.528823137283325 Tempo per calcolare la similarity matrix
        :param item_user_matrix:
        :param candidate_matrix:
        :return:
        """
        n_items = candidate_matrix.shape[0]
        # MULTIPROCESSING HERE
        for i in range(n_items):
            # Get the indices of the candidates for the i-th item
            candidate_indices = candidate_matrix.getrow(i).nonzero()[1]

            # Extract the relevant vectors from URM for these candidates
            URM_candidates = item_user_matrix[candidate_indices, :]

            # Compute cosine similarity between item i and its candidates
            item_vector = item_user_matrix[i, :]
            sim_scores = cosine_similarity(item_vector, URM_candidates)

            # Store the results
            self._similarity_matrix[i, candidate_indices] = sim_scores

    def get_user_recs(self, u, mask, k):
        user_id = self._data.public_users.get(u)
        user_recs = self._preds[user_id]
        # user_items = self._ratings[u].keys()
        user_recs_mask = mask[user_id]
        user_recs[~user_recs_mask] = -np.inf
        indices, values = zip(*[(self._data.private_items.get(u_list[0]), u_list[1])
                                for u_list in enumerate(user_recs)])

        # indices, values = zip(*predictions.items())
        indices = np.array(indices)
        values = np.array(values)
        local_k = min(k, len(values))
        partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
        real_values = values[partially_ordered_preds_indices]
        real_indices = indices[partially_ordered_preds_indices]
        local_top_k = real_values.argsort()[::-1]
        return [(real_indices[item], real_values[item]) for item in local_top_k]


    def get_model_state(self):
        saving_dict = {}
        saving_dict['_preds'] = self._preds
        saving_dict['_similarity'] = self._similarity
        saving_dict['_num_neighbors'] = self._num_neighbors
        saving_dict['_implicit'] = self._implicit
        return saving_dict

    def set_model_state(self, saving_dict):
        self._preds = saving_dict['_preds']
        self._similarity = saving_dict['_similarity']
        self._num_neighbors = saving_dict['_num_neighbors']
        self._implicit = saving_dict['_implicit']

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)
