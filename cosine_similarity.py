import numpy as np
import time


def compute_candidates_cosine_similarity_np(user_item_matrix: np.array,
                                            candidate_matrix: np.array):
    """
     In the case of Faiss or CustomLSH(my implementation of Faiss) we already retrieve the k candidates
     It makes no sense to sort them relatevely to the cosine similarity and so we skip this step and we output directly the components of the final sparse matrix
     :param user_item_matrix:
     :param item_user_matrix:
     :param candidate_matrix:
     :return:
    """
    dense_user_item_matrix = user_item_matrix
    candidates_users = dense_user_item_matrix[candidate_matrix]
    similarities = np.einsum("ijk,ik->ij", candidates_users, dense_user_item_matrix, optimize=True) / (np.sqrt(
        np.sum(candidates_users ** 2, axis=2)) * np.sqrt(
        np.sum(dense_user_item_matrix ** 2, axis=1)[:, np.newaxis]))
    return similarities
