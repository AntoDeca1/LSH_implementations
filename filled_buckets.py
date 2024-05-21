import numpy as np
from LSH_v4.LSH_v4 import RandomProjections
import time
from utils import create_sparse_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

"""
PARAMETERS
"""


def expected_non_empty_buckets_exact(n, nbits):
    num_buckets = 2 ** nbits
    return num_buckets * (1 - (1 - 1 / num_buckets) ** n)

def plot_comparison(seed, nbits, m, n_values, l, sparsity, trials):
    np.random.seed(seed)
    empirical_non_empty_buckets = []
    theoretical_non_empty_buckets_exact = [expected_non_empty_buckets_exact(n, nbits) for n in n_values]

    # Pre-generate sparse matrices for all trials and n_values
    sparse_matrices = {n_elem: [create_sparse_matrix(m, n_elem, sparsity) for _ in range(trials)] for n_elem in n_values}

    print(f"{'Number of elements (n)':<25} {'Empirical non-empty buckets':<30} {'Theoretical non-empty buckets'}")
    print('-' * 80)

    for n_elem in n_values:
        trial_non_empty_buckets = []
        for trial_matrix in sparse_matrices[n_elem]:
            rp = RandomProjections(d=m, l=l, nbits=nbits, seed=seed)
            rp.add(trial_matrix.T)
            filled_buckets = rp.all_hashes[0].shape[0]  # Count non-empty buckets
            trial_non_empty_buckets.append(filled_buckets)

        avg_non_empty_buckets = np.mean(trial_non_empty_buckets)
        empirical_non_empty_buckets.append(avg_non_empty_buckets)

        print(f"{n_elem:<25} {avg_non_empty_buckets:<30} {expected_non_empty_buckets_exact(n_elem, nbits)}")

    plt.plot(n_values, empirical_non_empty_buckets, marker='o', label='Empirical')
    plt.plot(n_values, theoretical_non_empty_buckets_exact, marker='x', linestyle='--', label='Theoretical (Exact)')
    plt.xlabel('Number of elements (n)')
    plt.ylabel('Number of non-empty buckets')
    plt.title(f'Number of non-empty buckets vs. Number of elements (nbits={nbits})')
    plt.legend()
    plt.grid(True)
    plt.show()


# Parameters
seed = 42
nbits = 10
m = 6400
n_values = [100, 1000, 2000, 3000, 4000, 5000]
l = 4
sparsity = 0.9
trials = 100  # Number of trials to smooth the curve

# Run the plot function
plot_comparison(seed, nbits, m, n_values, l, sparsity, trials)
