import numpy as np
from LSH_v2 import RandomProjections
from utils import create_sparse_matrix
import time
from sklearn.metrics.pairwise import cosine_similarity

"""
PARAMETERS
"""
seed = 42
nbits = 16  # number of hyperplanes in a three
m = 6400  # number of users
n = 3750  # number of items
l = 2  # number of thress in the forest
np.random.seed(seed)
"""
INPUT
"""
# user_item_matrix_dummy = np.random.randint(1, 5, size=(m, n))
user_item_matrix_dummy = create_sparse_matrix(m, n, sparsity=0.9)
"""
LSH Index
"""
rp = RandomProjections(d=m, l=2, nbits=nbits, seed=42)
"""
Index our vectors
"""
start = time.time()
rp.add(user_item_matrix_dummy.T)
end = time.time()
print("Time to index the vectors", end - start)
"""
Candidates
"""
# candidates_matrix = rp.candidates_matrix(k=5)
start = time.time()
similarities = cosine_similarity(user_item_matrix_dummy.T)
end = time.time()
print("Time to compute the itemXitem similarity",end-start)
"Similarities and Indexes"
start = time.time()
prova = rp.output_similarities_3()
end = time.time()
print("Time to compute the similarity with LSH", end - start)

# print("data", data)
# print("rows_indicies", rows_indices)
# print()
# print("cols_indptr", cols_indptr)
