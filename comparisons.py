import numpy as np
from utils import create_sparse_matrix, count_valid_indices
from sklearn.metrics.pairwise import cosine_similarity
import faiss

"""
PARAMETERS
Confrontare anche con l'implementazione di Faiss
"""
seed = 42
nbits = 128  # number of hyperplanes
m = 50  # number of users
n = 100  # number of items
np.random.seed(seed)
k = 20

user_item_matrix_dummy = create_sparse_matrix(m, n)

similarity_matrix = cosine_similarity(user_item_matrix_dummy.T)
for j in range(similarity_matrix.shape[0]):
    similarity_matrix[j][j] = -np.inf
candidates = np.argsort(similarity_matrix, axis=1)
candidates = [row[::-1] for row in candidates]
##LSH
item_user_matrix_dummy = user_item_matrix_dummy.T.toarray()
index = faiss.IndexLSH(m, nbits)
# Add the data to the index
index.add(item_user_matrix_dummy)
# print(D)  #Hamming distances to all the founded neighbours
candidates_v2 = None
for j in range(item_user_matrix_dummy.shape[0]):
    _, I = index.search(item_user_matrix_dummy[j, :].reshape(1, m), k=k)
    if candidates_v2 is None:
        candidates_v2 = I[0]
    else:
        candidates_v2 = np.vstack([candidates_v2, I[0]])

for i in range(candidates_v2.shape[0]):
    print("---------------------")
    query = item_user_matrix_dummy[i, :]
    closest_candidates = item_user_matrix_dummy[candidates[i][:k]]
    approximate_closest_candidates = item_user_matrix_dummy[candidates_v2[i, :]]
    print(candidates[i][:k])
    print(candidates_v2[i, :])
    print("------------AVG ORIGINAL COSINE SIMILARITY----------")
    print(np.mean(cosine_similarity(query.reshape(1, -1), closest_candidates)))
    print("------AVG APPROXIMATE COSINE SIMILARITY---------")
    print(np.mean(cosine_similarity(query.reshape(1, -1), approximate_closest_candidates)))
    print("-----------_ACCURACY-----------------------------")
    print(count_valid_indices(candidates_v2[i, :], candidates[i][:k]))
