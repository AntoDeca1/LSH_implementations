import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import random as sparse_random
import time
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def create_sparse_matrix_old(rows, cols, sparsity=0.7, seed=42):
    np.random.seed(seed)
    dense_matrix = np.random.randint(1, 6, size=(rows, cols))
    mask = np.random.choice([False, True], size=dense_matrix.shape, p=[sparsity, 1 - sparsity])
    sparse_matrix = dense_matrix * mask
    return csr_matrix(sparse_matrix)


def create_sparse_matrix(m, n, sparsity, seed=42):
    """Create a sparse matrix with a specified sparsity level."""
    np.random.seed(seed)
    return sparse_random(m, n, density=1 - sparsity, format='csr', random_state=42,
                         data_rvs=lambda size: np.random.randint(1, 5, size))


def time_cosine_similarity(m, n, sparsities, seed=42):
    # Determine which variable is a list and should be iterated over
    if isinstance(sparsities, list):
        variable = sparsities
        var_name = "Sparsity"
        fixed_params = f"m={m}, n={n}"
    elif isinstance(m, list):
        variable = m
        var_name = "Number of Users (m)"
        fixed_params = f"n={n}, Sparsity={sparsities}"
    elif isinstance(n, list):
        variable = n
        var_name = "Number of Items (n)"
        fixed_params = f"m={m}, Sparsity={sparsities}"
    else:
        raise ValueError("One of the parameters m, n, or sparsities must be a list.")

    times = []
    for val in variable:
        if var_name == "Number of Users (m)":
            m = val
        elif var_name == "Number of Items (n)":
            n = val
        elif var_name == "Sparsity":
            sparsities = val

        input_matrix = create_sparse_matrix(m, n, sparsities, seed)
        tic = time.time()
        cosine_similarity(input_matrix)
        elapsed_time = time.time() - tic
        times.append(elapsed_time)

    # Plotting for the specific parameter iteration
    plt.figure()
    plt.plot(variable, times, marker='o', label='Computation Time')
    plt.title(f'Cosine Similarity Computation Time for {fixed_params}')
    plt.xlabel(var_name)
    plt.ylabel('Time (seconds)')
    plt.grid(True)

    # Annotate percentage changes directly on the plot
    for i in range(1, len(times)):
        percentage_change = ((times[i] - times[i - 1]) / times[i - 1]) * 100
        plt.annotate(f'{percentage_change:.2f}%', (variable[i], times[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center')

    plt.show()

    # Create and display a DataFrame for better readability
    data = {
        var_name: variable,
        'Computation Time (seconds)': times
    }
    df = pd.DataFrame(data)
    print(df.to_string(index=False))


def stringify_array(array: np.array):
    return ''.join(map(str, array.flatten()))


def all_binary(n):
    total = 1 << n
    print(f"{total} possible combinations")
    combinations = []
    for i in range(total):
        # get binary representation of integer
        b = bin(i)[2:]
        # pad zeros to start of binary representtion
        b = '0' * (n - len(b)) + b
        b = [int(i) for i in b]
        combinations.append(b)
    return combinations


def initialize_random_matrix(d, k):
    return np.random.rand(d, k)


def create_mappings(buckets):
    """
    Given the buckets assigned for each vector(supposing vector are number from 0 to n-1)
    :param buckets:
    :return:
    """
    mapping_ = {}
    for i in range(buckets.shape[0]):
        stringified_bucket_id = stringify_array(buckets[i, :])
        if stringified_bucket_id in mapping_.keys():
            mapping_[stringified_bucket_id].append(i)
        else:
            mapping_[stringified_bucket_id] = [i]
    return mapping_


def hamming_distance_skratch(vec, buckets_hashes):
    hamming_dist = np.count_nonzero(vec != buckets_hashes, axis=1).reshape(-1, 1)
    # add hash values to each row
    hamming_dist = np.concatenate((projection.hashes, hamming_dist), axis=1)
    # sort based on distance
    hamming_dist = hamming_dist[hamming_dist[:, -1].argsort()]
    return hamming_dist
