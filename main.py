import numpy as np
from LSH import RandomProjections
from utils import create_buckets, create_mappings

nbits = 5  # number of hyperplanes
m = 50  # number of users
n = 100  # number of items
# Suppose this is the USER-INPUT MATRIX
# shape(user,item)
user_item_matrix_dummy = np.random.randint(1, 5, size=(m, n))
rp = RandomProjections(user_item_matrix_dummy.T, nbits)
rp.create_buckets()
similarities=rp.k_closest_matrix_2()
print()
