import numpy as np
from LSH_v2 import RandomProjections
from utils import create_sparse_matrix
import time
from sklearn.metrics.pairwise import cosine_similarity

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
np.random.seed(seed)
"""
INPUT
"""
# user_item_matrix_dummy = np.random.randint(1, 5, size=(m, n))
user_item_matrix_dummy = create_sparse_matrix(m, n, sparsity=sparsity)
"""
LSH Index
"""
rp = RandomProjections(d=m, l=l, nbits=nbits, seed=42)
"""
Index our vectors
"""
start = time.time()
rp.add(user_item_matrix_dummy.T)
end = time.time()
print("Time to index the vectors with LSH", end - start)
start = time.time()
similarities = cosine_similarity(user_item_matrix_dummy.T)
end = time.time()
print("Time to compute the itemXitem similarity", end - start)
"Similarities and Indexes"
start = time.time()
prova = rp.output_similarities_2(k=neighbours)
# prova=rp.output_similarities()
end = time.time()
print("Time to compute the similarity with LSH", end - start)

# POINTS TO IMPROVE
# SIMILARITY candidates_vector saturate when we have 20.000 vectors
# INDEXING TAKES TIME(precompute closest ) when we have a huge number of users but also a high number of nbits

# Increasing the number of bits mostly increase the indexing time(the search sometimes takes less time with output similarities2)
# DIFFERENT FROM OUTPUT_Similarity
# Try to swap nbits from 20 to 40
