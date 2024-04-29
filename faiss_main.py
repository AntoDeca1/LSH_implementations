import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import create_sparse_matrix
import time

"""
Utilizzo di Faiss puro.Un solo albero e possibilità di agire solo sul parametro nbits
"""

"""
PARAMETERS
"""
seed = 42
nbits = 8  # number of hyperplanes in a three --> Decrease the number of false positives
m = 6400  # number of users
n = 3750  # number of items
l = 4  # number of threes in the forest -->Decrease the number of false negatives
sparsity = 0.9
neighbours = 20
"""
INPUT
"""
user_item_matrix_dummy = create_sparse_matrix(m, n, sparsity=sparsity)
item_user_matrix_dummy = user_item_matrix_dummy.T.toarray()
# initialize the index using our vectors dimensionality (m) and nbits
index = faiss.IndexLSH(m, nbits)
# Add the data to the index
prima = time.time()
index.add(item_user_matrix_dummy)
dopo = time.time()
print(dopo - prima, "Time to index the vectors")
prima = time.time()
D, I = index.search(item_user_matrix_dummy, k=neighbours)
dopo = time.time()
print(dopo - prima, "Time to output the candidates")
print(I.shape)
print(D)
