import numpy as np
from LSH_v4.LSH_v4 import RandomProjections
from LSH_faiss.LSH_faiss import LSH
from utils import create_sparse_matrix
import time
import matplotlib.pyplot as plt
import pandas as pd


# Function to calculate theoretical indexing time
def theoretical_indexing_time(l, n, d, nbits):
    return l * n * d * nbits


# Function to calculate theoretical search time
def theoretical_search_time(n, l, nbits, k):
    term1 = n * l * nbits * 2 ** nbits * (1 - (1 - 1 / 2 ** nbits) ** n)
    term2 = (k ** 2 * 2 ** nbits) / l
    return term1 + term2


# Function to run the tests and collect timing data
def run_tests(seed, nbits_list, m_list, n_list, l_list, sparsity, neighbours):
    results = []

    for nbits in nbits_list:
        for m in m_list:
            for n in n_list:
                for l in l_list:
                    np.random.seed(seed)

                    # Create sparse matrix
                    user_item_matrix_dummy = create_sparse_matrix(m, n, sparsity=sparsity)

                    # Initialize LSH
                    rp = RandomProjections(d=m, l=l, nbits=nbits, seed=seed)

                    # Index vectors
                    start = time.time()
                    rp.add(user_item_matrix_dummy.T)
                    end = time.time()
                    index_time = end - start

                    # Retrieve candidates
                    start = time.time()
                    rp.search_2(k=neighbours)
                    end = time.time()
                    search_time = end - start

                    # Calculate theoretical times
                    theoretical_index_time = theoretical_indexing_time(l, n, m, nbits)
                    theoretical_search_time_value = theoretical_search_time(n, l, nbits, neighbours)

                    # Store results
                    results.append({
                        'nbits': nbits,
                        'm': m,
                        'n': n,
                        'l': l,
                        'index_time': index_time,
                        'search_time': search_time,
                        'theoretical_index_time': theoretical_index_time,
                        'theoretical_search_time': theoretical_search_time_value
                    })

    return pd.DataFrame(results)


# Parameters
seed = 42
sparsity = 0.9
neighbours = 200

# Define ranges for the parameters to test
nbits_list = [10, 15, 20]  # Different number of hyperplanes
m_list = [3200, 6400, 12800]  # Different number of users
n_list = [2000, 4000, 8000]  # Different number of items
l_list = [2, 4, 8]  # Different number of hash tables

# Run tests
results_df = run_tests(seed, nbits_list, m_list, n_list, l_list, sparsity, neighbours)

# Normalize the times
results_df['index_time'] = results_df['index_time'] / results_df['index_time'].max()
results_df['search_time'] = results_df['search_time'] / results_df['search_time'].max()
results_df['theoretical_index_time'] = results_df['theoretical_index_time'] / results_df['theoretical_index_time'].max()
results_df['theoretical_search_time'] = results_df['theoretical_search_time'] / results_df[
    'theoretical_search_time'].max()

# Plot the results
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# Plot indexing time
axes[0].plot(results_df.index, results_df['index_time'], label='Actual Index Time', marker='o')
axes[0].plot(results_df.index, results_df['theoretical_index_time'], label='Theoretical Index Time', marker='x')
axes[0].set_title('Normalized Indexing Time')
axes[0].set_xlabel('Test Case Index')
axes[0].set_ylabel('Normalized Time')
axes[0].legend()

# Plot search time
axes[1].plot(results_df.index, results_df['search_time'], label='Actual Search Time', marker='o')
axes[1].plot(results_df.index, results_df['theoretical_search_time'], label='Theoretical Search Time', marker='x')
axes[1].set_title('Normalized Search Time')
axes[1].set_xlabel('Test Case Index')
axes[1].set_ylabel('Normalized Time')
axes[1].legend()

plt.tight_layout()
plt.show()
