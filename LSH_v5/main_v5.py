import numpy as np
from LSH_v5 import RandomProjections
from utils import create_sparse_matrix
import time
from sklearn.metrics.pairwise import cosine_similarity

"""
PARAMETERS
"""
seed = 42
nbits = 256 # number of hyperplanes in a three --> Decrease the number of false positives
m = 5900  # number of users
n = 3750  # number of items
l = 1  # number of threes in the forest -->Decrease the number of false negatives
sparsity = 0.97
neighbours = 200
np.random.seed(seed)
"""
INPUT
"""
user_item_matrix_dummy = create_sparse_matrix(m, n, sparsity=sparsity)
"""
LSH Index
"""
rp = RandomProjections(d=m, nbits=nbits, seed=42)
"""
Index our vectors
"""
start = time.time()
rp.add(user_item_matrix_dummy.T)
end = time.time()
print("Time to index the vectors with LSH", end - start)
# start = time.time()
# similarities = cosine_similarity(user_item_matrix_dummy.T)
# end = time.time()
print("Time to compute the itemXitem similarity", end - start)
"Similarities and Indexes"
start = time.time()
rp.search_2(k=neighbours)
end = time.time()
print("Time to compute the retrieve the candidates with LSH", end - start)
