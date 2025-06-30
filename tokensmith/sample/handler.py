from typing import TYPE_CHECKING, Union, List, Dict, Tuple, Optional, Any
import numpy as np
from ..utils import generate_training_sample

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = Any

if TYPE_CHECKING:
    from ..manager import DatasetManager  # only used for type hints, won't cause import loop

class SampleHandler:
    def __init__(self, manager: 'DatasetManager'):
        self.manager = manager

    def get_samples_by_indices(
        self, 
        indices: List[int], 
        return_doc_details: bool = False,
        return_detokenized: bool = False,
        tokenizer: Optional[Any] = None,
    ) -> Union[List[List[np.ndarray]], List[str], List[Tuple[List[np.ndarray], Dict]], List[Tuple[str, Dict]]]:
        """
        Returns a list of samples by their indices, optionally with document details and/or detokenized.

        Parameters:
            indices (List[int]): List of sample indices to retrieve.
            return_doc_details (bool): If True, includes associated document details.
            return_detokenized (bool): If True, returns detokenized text instead of token arrays.
            tokenizer: The tokenizer to use for detokenization (required if return_detokenized is True).

        Raises:
            ValueError: If indices is not a list of non-negative integers or if tokenizer is None when return_detokenized is True.

        Returns:
            List[List[np.ndarray]]: A list of samples, where each sample is a list of token arrays (if return_detokenized is False and return_doc_details is False).
            List[str]: A list of detokenized text samples (if return_detokenized is True and return_doc_details is False).
            List[Tuple[List[np.ndarray], Dict]]: A list of tuples containing token sequences and document details (if return_detokenized is False and return_doc_details is True).
            List[Tuple[str, Dict]]: A list of tuples containing detokenized text and document details (if return_detokenized is True and return_doc_details is True).
        """
        if not isinstance(indices, list):
            raise ValueError("indices must be a list.")
        if not all(isinstance(i, int) for i in indices):
            raise ValueError("All elements in indices must be integers.")
        if not all(i >= 0 for i in indices):
            raise ValueError("All elements in indices must be non-negative integers.")

        if return_detokenized and tokenizer is None:
            raise ValueError("tokenizer must be provided if return_detokenized is True.")

        samples = []
        for index in indices:
            response = self.manager.WriteableMMapIndexedDataset.get_example_by_id(
                example_loc=index,
                return_doc_details=return_doc_details
            )

            if return_doc_details:
                output_seq, doc_details = response
            else:
                output_seq = response
                doc_details = None

            if return_detokenized:
                output_seq = generate_training_sample(output_seq, tokenizer)

            if return_doc_details:
                samples.append((output_seq, doc_details))
            else:
                samples.append(output_seq)

        return samples

    def get_batches_by_ids(
        self,
        batch_ids: List[int],
        batch_size: int,
        return_doc_details: bool = False,
        return_detokenized: bool = False,
        tokenizer: Optional[Any] = None,
    ) -> Union[List[List[List[np.ndarray]]], List[List[str]], List[List[Tuple[List[np.ndarray], Dict]]], List[List[Tuple[str, Dict]]]]:
        """
        Returns samples from multiple batches by their batch IDs, organized by batch, optionally with document details and/or detokenized.

        Parameters:
            batch_ids (List[int]): List of batch IDs to retrieve.
            batch_size (int): The size of each batch.
            return_doc_details (bool): If True, includes associated document details.
            return_detokenized (bool): If True, returns detokenized text instead of token arrays.
            tokenizer: The tokenizer to use for detokenization (required if return_detokenized is True).

        Raises:
            ValueError: If batch_ids is not a list of non-negative integers or if tokenizer is None when return_detokenized is True.

        Returns:
            List[List[List[np.ndarray]]]: A list of batches, where each batch is a list of samples (if return_detokenized is False and return_doc_details is False).
            List[List[str]]: A list of batches, where each batch is a list of detokenized text samples (if return_detokenized is True and return_doc_details is False).
            List[List[Tuple[List[np.ndarray], Dict]]]: A list of batches with token sequences and document details (if return_detokenized is False and return_doc_details is True).
            List[List[Tuple[str, Dict]]]: A list of batches with detokenized text and document details (if return_detokenized is True and return_doc_details is True).
        """
        if not isinstance(batch_ids, list):
            raise ValueError("batch_ids must be a list.")
        if not all(isinstance(i, int) for i in batch_ids):
            raise ValueError("All elements in batch_ids must be integers.")
        if not all(i >= 0 for i in batch_ids):
            raise ValueError("All elements in batch_ids must be non-negative integers.")

        if return_detokenized and tokenizer is None:
            raise ValueError("tokenizer must be provided if return_detokenized is True.")

        # Collect samples organized by batch
        batches = []
        for batch_id in batch_ids:
            batch_indices = [i for i in range(batch_id * batch_size, (batch_id + 1) * batch_size)]
            batch_samples = self.get_samples_by_indices(
                indices=batch_indices,
                return_doc_details=return_doc_details,
                return_detokenized=return_detokenized,
                tokenizer=tokenizer
            )
            batches.append(batch_samples)

        return batches

    def get_samples_by_policy(
        self,
        policy_fn: callable,
        *policy_args,
        return_doc_details: bool = False,
        return_detokenized: bool = False,
        tokenizer: Optional[Any] = None,
        **policy_kwargs
    ) -> Union[List[List[np.ndarray]], List[str], List[Tuple[List[np.ndarray], Dict]], List[Tuple[str, Dict]]]:
        """
        Returns samples based on a sampling policy function that generates indices.

        Parameters:
            policy_fn (callable): A function that returns a list of sample indices.
            *policy_args: Positional arguments to pass to the policy function.
            return_doc_details (bool): If True, includes associated document details.
            return_detokenized (bool): If True, returns detokenized text instead of token arrays.
            tokenizer: The tokenizer to use for detokenization (required if return_detokenized is True).
            **policy_kwargs: Keyword arguments to pass to the policy function.

        Raises:
            ValueError: If policy_fn is not callable or doesn't return a list of integers.

        Returns:
            List[List[np.ndarray]]: A list of samples, where each sample is a list of token arrays (if return_detokenized is False and return_doc_details is False).
            List[str]: A list of detokenized text samples (if return_detokenized is True and return_doc_details is False).
            List[Tuple[List[np.ndarray], Dict]]: A list of tuples containing token sequences and document details (if return_detokenized is False and return_doc_details is True).
            List[Tuple[str, Dict]]: A list of tuples containing detokenized text and document details (if return_detokenized is True and return_doc_details is True).
        """
        if not callable(policy_fn):
            raise ValueError("policy_fn must be callable.")

        indices = policy_fn(*policy_args, **policy_kwargs)

        if not isinstance(indices, list):
            raise ValueError("policy_fn must return a list of integers.")

        return self.get_samples_by_indices(
            indices=indices,
            return_doc_details=return_doc_details,
            return_detokenized=return_detokenized,
            tokenizer=tokenizer
        )

    def get_batches_by_policy(
        self,
        policy_fn: callable,
        batch_size: int,
        *policy_args,
        return_doc_details: bool = False,
        return_detokenized: bool = False,
        tokenizer: Optional[Any] = None,
        **policy_kwargs
    ) -> Union[List[List[List[np.ndarray]]], List[List[str]], List[List[Tuple[List[np.ndarray], Dict]]], List[List[Tuple[str, Dict]]]]:
        """
        Returns batches of samples based on a sampling policy function that generates batch IDs.

        Parameters:
            policy_fn (callable): A function that returns a list of batch IDs.
            batch_size (int): The size of each batch.
            *policy_args: Positional arguments to pass to the policy function.
            return_doc_details (bool): If True, includes associated document details.
            return_detokenized (bool): If True, returns detokenized text instead of token arrays.
            tokenizer: The tokenizer to use for detokenization (required if return_detokenized is True).
            **policy_kwargs: Keyword arguments to pass to the policy function.

        Raises:
            ValueError: If policy_fn is not callable or doesn't return a list of integers.

        Returns:
            List[List[List[np.ndarray]]]: A list of batches, where each batch is a list of samples (if return_detokenized is False and return_doc_details is False).
            List[List[str]]: A list of batches, where each batch is a list of detokenized text samples (if return_detokenized is True and return_doc_details is False).
            List[List[Tuple[List[np.ndarray], Dict]]]: A list of batches with token sequences and document details (if return_detokenized is False and return_doc_details is True).
            List[List[Tuple[str, Dict]]]: A list of batches with detokenized text and document details (if return_detokenized is True and return_doc_details is True).
        """
        if not callable(policy_fn):
            raise ValueError("policy_fn must be callable.")

        batch_ids = policy_fn(*policy_args, **policy_kwargs)

        if not isinstance(batch_ids, list):
            raise ValueError("policy_fn must return a list of integers.")

        batches = []
        for batch_id in batch_ids:
            # Get indices for this single batch
            indices = [i for i in range(batch_id * batch_size, (batch_id + 1) * batch_size)]
            batch = self.get_samples_by_indices(
                indices=indices,
                return_doc_details=return_doc_details,
                return_detokenized=return_detokenized,
                tokenizer=tokenizer
            )
            batches.append(batch)

        return batches