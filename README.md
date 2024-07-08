# LSH Implementations for Thesis Work

## Introduction

This repository contains all the Locality-Sensitive Hashing (LSH) implementations developed during my thesis work. The primary aim is to investigate whether LSH random projections can be implemented in Python while providing tangible advantages. The project explores various approaches to LSH implementation, comparing custom Python solutions with established libraries.

## Implementations

### LSH_faiss

This directory contains a simple wrapper class around the Faiss LSH index. The core logic is implemented by Faiss, a powerful library for efficient similarity search. The Faiss index was encapsulated in a custom class to provide a common interface across all implementations in this project.
### LSH_v3
LSH_v3 is an implementation that follows the original RandomProjections idea. Unlike the Faiss-inspired approaches, LSH_v3 doesn't search for candidates by Hamming distance. Instead, it considers as candidates only the vectors that fall exactly in the same bucket as the query vector. This approach results in faster candidate retrieval but at the cost of losing control over the actual number of candidates. Consequently, the results from this implementation need to be interpreted differently from the other versions. While potentially faster, the trade-off is in the variability of the candidate set size and potentially reduced recall for certain queries.

### LSH_v4

LSH_v4 is a custom implementation that follows the ideas presented in PineCone articles. It serves as a first Python-based counterpart to the Faiss LSH approach
### LSH_faiss_likemp
The idea is similar to the one proposed by Faiss. That is, the search is done exhaustively in the hamming space.Not being able to use multithreading as Faiss does due to language limitations, multiprocessing was used. 

### LSH_v5

LSH_v5 is an improvement upon LSH_v4. While the core logic remains the same, this version introduces multiprocessing in the candidates retrieval phase. This enhancement aims to leverage parallel processing capabilities to speed up the retrieval of candidate vectors.


## Usage
Each implementation (LSH_faiss, LSH_v4, LSH_v5, and LSH_v6) has its own main file with timing functions. These files are designed to facilitate testing and benchmarking of the respective implementations. To use any of the implementations:

1. Navigate to the desired implementation directory.
2. Locate the main file (e.g., `main_faiss.py`, `main_v4.py`, etc.).
3. Run the main file to execute the implementation with built-in timing functions.

Example:

cd LSH_v4

python main_v4.py
These main files provide a convenient way to test the performance and functionality of each LSH implementation.

## Results and Comparisons

The project is currently in progress, and comprehensive results are still being compiled and analyzed. Detailed performance comparisons and analysis will be made available as soon as they are formalized.

## Future Work

TBD

## Contributing

TBD

## License

TBD
