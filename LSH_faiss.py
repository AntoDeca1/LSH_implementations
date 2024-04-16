import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import create_sparse_matrix

"""
PARAMETERS
"""
nbits = 5  # number of hyperplanes
m = 50  # number of users
n = 100  # number of items
np.random.seed(42)
"""
INPUT
"""
# user_item_matrix_dummy = np.random.randint(1, 5, size=(m, n))
user_item_matrix_dummy = create_sparse_matrix(m, n)
item_user_matrix_dummy = user_item_matrix_dummy.T.toarray()
# initialize the index using our vectors dimensionality (m) and nbits
index = faiss.IndexLSH(m, nbits)
# Add the data to the index
index.add(item_user_matrix_dummy)
D, I = index.search(item_user_matrix_dummy[0, :].reshape(1, m), k=5)
# print(D)  #Hamming distances to all the founded neighbours
print(I[0])
print(item_user_matrix_dummy[I[0]])
similarities = cosine_similarity(item_user_matrix_dummy[I[0]], [item_user_matrix_dummy[1, :]])
print(similarities)
