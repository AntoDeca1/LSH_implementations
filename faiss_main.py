import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import create_sparse_matrix
import time
from LSH_faiss import LSH_faiss

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
# # user_item_matrix_dummy = np.random.randint(1, 5, size=(m, n))
user_item_matrix_dummy = create_sparse_matrix(m, n, sparsity=sparsity)
item_user_matrix_dummy = user_item_matrix_dummy.T.toarray()
# # initialize the index using our vectors dimensionality (m) and nbits
# prima = time.time()
# index = faiss.IndexLSH(m, nbits)
# # Add the data to the index
# index.add(item_user_matrix_dummy)
# D, I = index.search(item_user_matrix_dummy[0, :].reshape(1, m), k=neighbours)
# # print(D)  #Hamming distances to all the founded neighbours
# print(I[0])
# print(item_user_matrix_dummy[I[0]])
# similarities = cosine_similarity(item_user_matrix_dummy[I[0]], [item_user_matrix_dummy[1, :]])
# dopo = time.time()
# print(dopo - prima)
# print(similarities)

prima = time.time()
index = LSH_faiss(m, l, nbits)
index.add(item_user_matrix_dummy)
index.search(k=neighbours)
dopo = time.time()
print(dopo - prima)
