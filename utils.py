import numpy as np
from scipy.sparse import csr_matrix


def create_sparse_matrix(rows, cols, sparsity=0.7, seed=42):
    np.random.seed(seed)
    dense_matrix = np.random.randint(1, 6, size=(rows, cols))
    mask = np.random.choice([False, True], size=dense_matrix.shape, p=[sparsity, 1 - sparsity])
    sparse_matrix = dense_matrix * mask
    return csr_matrix(sparse_matrix)

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
