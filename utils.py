import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import random as sparse_random
import time
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def create_sparse_matrix(m, n, sparsity, seed=42):
    """Create a sparse matrix with a specified sparsity level."""
    np.random.seed(seed)
    return sparse_random(m, n, density=1 - sparsity, format='csr', random_state=42,
                         data_rvs=lambda size: np.random.randint(1, 5, size))


def stringify_array(array: np.array):
    """
    Useful to transform a binary array in a stringified version
    Used in LSH_v4 for creating a dictionary key starting from a vector
    :param array:
    :return:
    """
    return ''.join(map(str, array.flatten()))


def hamming_distance(a, b):
    """
    Compute the hamming distance between two binary vectors
    :param a:
    :param b:
    :return:
    """
    return np.count_nonzero(a != b)


