import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import create_sparse_matrix
import time
from LSH_faiss import LSH

"""
PARAMETERS
"""


seed = 42
nbits = 8  # number of hyperplanes in a three --> Decrease the number of false positives
m = 6400  # number of users
n = 10000  # number of items
sparsity = 0.9
neighbours = 20
"""
INPUT
"""
user_item_matrix_dummy = create_sparse_matrix(m, n, sparsity=sparsity, seed=seed)
item_user_matrix_dummy = user_item_matrix_dummy.T.toarray()
# initialize the index using our vectors dimensionality (m) and nbits
index = LSH(m, nbits)
# Index the vectors
prima = time.time()
index.add(item_user_matrix_dummy)
dopo = time.time()
print(dopo - prima, "Tempo per indicizzare i vettori")
# Search for candidates
prima = time.time()
index.search(item_user_matrix_dummy, k=20)
dopo = time.time()
print(dopo - prima, "tempo per restituire i candidati in formato sparso n_itemsXn_items")
prima = time.time()
index.search_2(item_user_matrix_dummy, k=20)
dopo = time.time()
print(dopo - prima, "tempo per restituire i candidati come indici in un ndarray n_itemsXk")
