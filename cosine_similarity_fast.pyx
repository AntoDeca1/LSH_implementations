# cosine_similarity.pyx
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
from libc.math cimport sqrt
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

def compute_candidates_cosine_similarity_np(np.ndarray[np.float64_t, ndim=2] user_item_matrix,
                                            np.ndarray[np.int64_t, ndim=2] candidate_matrix):
    """
    Calculate cosine similarity for candidate matrices.
    """
    cdef:
        int num_users = user_item_matrix.shape[0]
        int num_features = user_item_matrix.shape[1]
        int num_candidates = candidate_matrix.shape[0]
        int i, j, k
        double sum_sq_u, sum_sq_c, dot_product
        np.ndarray[np.float64_t, ndim=2] similarities = np.zeros((num_candidates, num_users), dtype=np.float64)

    for i in range(num_candidates):
        for j in range(num_users):
            dot_product = 0.0
            sum_sq_u = 0.0
            sum_sq_c = 0.0
            for k in range(num_features):
                dot_product += user_item_matrix[candidate_matrix[i, j], k] * user_item_matrix[j, k]
                sum_sq_u += user_item_matrix[j, k] ** 2
                sum_sq_c += user_item_matrix[candidate_matrix[i, j], k] ** 2

            sum_sq_u = sqrt(sum_sq_u)
            sum_sq_c = sqrt(sum_sq_c)
            if sum_sq_u == 0 or sum_sq_c == 0:
                similarities[i, j] = 0
            else:
                similarities[i, j] = dot_product / (sum_sq_u * sum_sq_c)

    return similarities
