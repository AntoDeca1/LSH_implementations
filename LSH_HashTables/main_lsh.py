import numpy as np
from LSH_HashTables.LSH import LSHRp
from utils import create_sparse_matrix
import time

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
index = LSHRp(dim=m, num_tables=l, hash_size=nbits)
prima = time.time()
index.add(user_item_matrix_dummy.T)
dopo = time.time()
print(dopo - prima, "Tempo per indicizzare i vettori")
prima = time.time()
ds = index.query(user_item_matrix_dummy.T)
dopo = time.time()
print(dopo-prima,"Tempo per tirar fuori i candidati")
