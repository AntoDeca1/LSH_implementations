import time

from utils import create_sparse_matrix, time_cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

seed = 42
m_values = [6000, 7000, 8000, 9000, 10000, 20000]  # number of users
n_values = [3000, 5000, 7000, 9000, 11000]  # number of items
sparsities = [0.9, 0.95, 0.97, 0.98]

# Assuming you want to collect times for a single combination of m and n for each sparsity level
time_cosine_similarity(m=m_values, n=n_values[0], sparsities=sparsities[3])
