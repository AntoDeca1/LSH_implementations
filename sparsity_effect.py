import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from utils import create_sparse_matrix
from LSH_faiss.LSH_faiss import LSH


def baseline(user_item_matrix):
    """
    In the case of Faiss or CustomLSH(my implementation of Faiss) we already retrieve the k candidates
    It makes no sense to sort them relatively to the cosine similarity and so we skip this step and we output directly the components of the final sparse matrix
    :param item_user_matrix:
    :param candidate_matrix:
    :return:
    """
    n_users = user_item_matrix.shape[0]
    similarity_matrix = np.empty((n_users, n_users))
    for i in range(n_users):
        # Compute cosine similarity between item i and its candidates
        user_vector = user_item_matrix[i]
        sim_scores = cosine_similarity(user_vector, user_item_matrix)
        similarity_matrix[i] = sim_scores
    return similarity_matrix


seed = 42
nbits = 256  # number of hyperplanes in a three --> Decrease the number of false positives
m = 5900  # number of users
n = 3750  # number of items
l = 3  # number of threes in the forest -->Decrease the number of false negatives
sparsities = [0.97, 0.98, 0.99, 0.9996]
neighbours = 200
num_trials = 10
np.random.seed(seed)

times = []
indexing_times = []
search_times = []

for sparsity in sparsities:
    baseline_times = []
    indexing_times_trial = []
    search_times_trial = []
    user_item_matrix_dummy = create_sparse_matrix(m, n, sparsity=sparsity)

    for _ in range(num_trials):
        before = time.time()
        baseline(user_item_matrix_dummy)
        time_needed = time.time() - before
        baseline_times.append(time_needed)

        index = LSH(n, nbits)

    avg_baseline_time = np.mean(baseline_times)

    times.append(avg_baseline_time)
    print(f"----------Sparsity: {sparsity}----------")
    print(f"Average Baseline time needed: {avg_baseline_time}s")
# Plotting the results
plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.plot(sparsities, times, marker='o')
plt.xlabel('Sparsity')
plt.ylabel('Baseline Time (seconds)')
plt.title('Baseline Time Needed vs. Sparsity')
plt.grid(True)

plt.tight_layout()
plt.show()
