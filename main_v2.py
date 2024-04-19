import numpy as np
from LSH_v2 import RandomProjections
from utils import create_sparse_matrix
import time

"""
PARAMETERS
"""
seed = 42
nbits = 5  # number of hyperplanes in a three
m = 6040  # number of users
n = 3675  # number of items
l = 10  # number of thress in the forsest
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
rp.add(user_item_matrix_dummy.T)
"""
Candidates
"""
# candidates_matrix = rp.candidates_matrix(k=5)
"Similarities and Indexes"
start = time.time()
prova = rp.output_similarities()
end = time.time()
print(end - start)
# print("data", data)
# print("rows_indicies", rows_indices)
# print()
# print("cols_indptr", cols_indptr)
