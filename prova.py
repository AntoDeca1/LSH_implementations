from cosine_similarity_fast import compute_candidates_cosine_similarity_np
from cosine_similarity import compute_candidates_cosine_similarity_np as csnp
import numpy as np
import time
# Example usage
user_item_matrix = np.random.rand(10, 5)
candidate_matrix = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
prima = time.time()
similarities = compute_candidates_cosine_similarity_np(user_item_matrix, candidate_matrix)
print(time.time() - prima)
prima = time.time()
similarities = csnp(user_item_matrix, candidate_matrix)
print(time.time() - prima)
print(similarities)
