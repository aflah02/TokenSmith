# Heavily inspired by the original code from https://github.com/EleutherAI/tokengrams/blob/master/tokengrams/tokengrams.pyi and uses the same library.

from tokengrams import MemmapIndex
from typing import List
import os
import logging

class SearchHandler:
    def __init__(self, bin_file_path: str, index_save_path: str, vocab: int, verbose: bool = True, reuse: bool = True):

        self.bin_file_path = bin_file_path
        self.index_save_path = index_save_path
        self.vocab = vocab
        self.verbose = verbose
        self.reuse = reuse

        if vocab not in [2**16, 2**32]:
            raise ValueError("vocab must be either 2**16 or 2**32. Set it to 2**16 if your token vocabulary is less than 2**16, or 2**32 if it is larger than that.")

        if reuse:
            if os.path.exists(os.path.join(self.index_save_path)):
                logging.info("Reusing existing index.")
                self.index = MemmapIndex(
                    self.bin_file_path,
                    self.index_save_path,
                    vocab=self.vocab,
                )
            else:
                raise ValueError(f"Index path {self.index_save_path} does not exist. Set reuse=False to create a new index.")
        else:
            logging.info("Creating a new index.")

            self.index = MemmapIndex.build(
                self.bin_file_path,
                self.index_save_path,
                vocab=self.vocab,
                verbose=self.verbose,
            )

        assert self.index.is_sorted(), "The index is not sorted. This is not expected. Please rerun the index creation process."

    def count(self, query: List[int]) -> int:
        """Counts the occurrences of a query in the index."""
        if not isinstance(query, list):
            raise ValueError("query must be a list of integers.")
        if not all(isinstance(token, int) for token in query):
            raise ValueError("All elements in query must be integers.")
        if len(query) == 0:
            raise ValueError("query cannot be an empty list.")
        return self.index.count(query)

    def contains(self, query: List[int]) -> bool:
        """Checks if a query is present in the index."""
        if not isinstance(query, list):
            raise ValueError("query must be a list of integers.")
        if not all(isinstance(token, int) for token in query):
            raise ValueError("All elements in query must be integers.")
        if len(query) == 0:
            raise ValueError("query cannot be an empty list.")
        return self.index.contains(query)

    def positions(self, query: List[int]) -> List[int]:
        """Returns an unordered list of positions where `query` starts in `tokens`."""
        if not isinstance(query, list):
            raise ValueError("query must be a list of integers.")
        if not all(isinstance(token, int) for token in query):
            raise ValueError("All elements in query must be integers.")
        if len(query) == 0:
            raise ValueError("query cannot be an empty list.")
        return self.index.positions(query)

    def count_next(self, query: List[int]) -> List[int]:
        """Count the occurrences of each token directly following `query`."""
        if not isinstance(query, list):
            raise ValueError("query must be a list of integers.")
        if not all(isinstance(token, int) for token in query):
            raise ValueError("All elements in query must be integers.")
        if len(query) == 0:
            raise ValueError("query cannot be an empty list.")
        return self.index.count_next(query)

    def batch_count_next(self, queries: List[List[int]]) -> List[List[int]]:
        """Count the occurrences of each token directly following each query in a batch."""
        if not isinstance(queries, list):
            raise ValueError("queries must be a list of lists of integers.")
        if not all(isinstance(query, list) for query in queries):
            raise ValueError("All elements in queries must be lists of integers.")
        if not all(all(isinstance(token, int) for token in query) for query in queries):
            raise ValueError("All elements in queries must be integers.")
        if any(len(query) == 0 for query in queries):
            raise ValueError("None of the queries can be an empty list.")
        return self.index.batch_count_next(queries)

    def sample_smoothed(self, query: List[int], n: int, k: int, num_samples: int) -> List[List[int]]:
        """Sample `num_samples` sequences of length `k` that follow `query` based on previous (n- 1) characters (n-gram prefix). Uses a Kneser-New smoothed conditional distribution. If less than (n - 1) characters are available, it uses all available characters."""
        if not isinstance(query, list):
            raise ValueError("query must be a list of integers.")
        if not all(isinstance(token, int) for token in query):
            raise ValueError("All elements in query must be integers.")
        if len(query) == 0:
            raise ValueError("query cannot be an empty list.")
        return self.index.sample_smoothed(query, n, k, num_samples)

    def sample_unsmoothed(self, query: List[int], k: int, num_samples: int) -> List[List[int]]:
        """Sample `num_samples` sequences of length `k` that follow `query` based on previous characters (n-gram prefix). If less than (n - 1) characters are available, it uses all available characters."""
        if not isinstance(query, list):
            raise ValueError("query must be a list of integers.")
        if not all(isinstance(token, int) for token in query):
            raise ValueError("All elements in query must be integers.")
        if len(query) == 0:
            raise ValueError("query cannot be an empty list.")
        return self.index.sample_unsmoothed(query, k, num_samples)

    def get_smoothed_probs(self, query: List[int], k: int) -> List[float]:
        """Get the interpolated Kneser-Ney smoothed token probability distribution using all previous tokens in the query."""
        if not isinstance(query, list):
            raise ValueError("query must be a list of integers.")
        if not all(isinstance(token, int) for token in query):
            raise ValueError("All elements in query must be integers.")
        if len(query) == 0:
            raise ValueError("query cannot be an empty list.")
        return self.index.get_smoothed_probs(query, k)
    
    def batch_get_smoothed_probs(self, queries: List[List[int]], k: int) -> List[List[float]]:
        """Get the interpolated Kneser-Ney smoothed token probability distribution using all previous tokens in each query."""
        if not isinstance(queries, list):
            raise ValueError("queries must be a list of lists of integers.")
        if not all(isinstance(query, list) for query in queries):
            raise ValueError("All elements in queries must be lists of integers.")
        if not all(all(isinstance(token, int) for token in query) for query in queries):
            raise ValueError("All elements in queries must be integers.")
        if any(len(query) == 0 for query in queries):
            raise ValueError("None of the queries can be an empty list.")
        return self.index.batch_get_smoothed_probs(queries, k)

    def estimate_delta(self, n: int) -> None:
        """Warning: O(k**n) where k is vocabulary size, use with caution.
        Improve smoothed model quality by replacing the default delta hyperparameters
        for models of order n and below with improved estimates over the entire index.
        https://people.eecs.berkeley.edu/~klein/cs294-5/chen_goodman.pdf, page 16."""
        if not isinstance(n, int):
            raise ValueError("n must be an integer.")
        self.index.estimate_delta(n)