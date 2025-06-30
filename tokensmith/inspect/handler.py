from typing import TYPE_CHECKING, Union, List, Dict, Tuple, Optional, Any
import numpy as np
from ..utils import generate_training_sample

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = Any

if TYPE_CHECKING:
    from ..manager import DatasetManager  # only used for type hints, won't cause import loop

class InspectHandler:
    def __init__(self, manager: 'DatasetManager'):
        self.manager = manager

    def inspect_sample_by_id(
        self,
        sample_id: int,
        return_doc_details: bool = False,
        return_detokenized: bool = False,
        tokenizer: Optional[Any] = None,
    ) -> Union[List[np.ndarray], str, Tuple[List[np.ndarray], Dict], Tuple[str, Dict]]:
        """
        Returns a sample by its ID, optionally with document details and/or detokenized.

        Parameters:
            sample_id (int): The index of the sample to retrieve.
            return_doc_details (bool): If True, includes associated document details.
            return_detokenized (bool): If True, returns detokenized text instead of token arrays.
            tokenizer: The tokenizer to use for detokenization (required if return_detokenized is True).

        Raises:
            ValueError: If sample_id is not a non-negative integer or if tokenizer is None when return_detokenized is True.

        Returns:
            List[np.ndarray]: A list of numpy arrays representing the token sequence (if return_detokenized is False and return_doc_details is False).
            str: Detokenized text (if return_detokenized is True and return_doc_details is False).
            Tuple[List[np.ndarray], Dict]: Token sequence and document details (if return_detokenized is False and return_doc_details is True).
            Tuple[str, Dict]: Detokenized text and document details (if return_detokenized is True and return_doc_details is True).
        """
        if sample_id < 0:
            raise ValueError("sample_id must be a non-negative integer.")

        if return_detokenized and tokenizer is None:
            raise ValueError("tokenizer must be provided if return_detokenized is True.")

        response = self.manager.WriteableMMapIndexedDataset.get_example_by_id(
            example_loc=sample_id,
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
            return output_seq, doc_details
        else:
            return output_seq

    def inspect_sample_by_batch(
        self,
        batch_id: int,
        batch_size: int,
        return_doc_details: bool = False,
        return_detokenized: bool = False,
        tokenizer: Optional[Any] = None,
    ) -> Union[List[List[np.ndarray]], List[str], List[Tuple[List[np.ndarray], Dict]], List[Tuple[str, Dict]]]:
        """
        Returns a batch of samples by batch ID, optionally with document details and/or detokenized.

        Parameters:
            batch_id (int): The index of the batch to retrieve.
            batch_size (int): The size of the batch.
            return_doc_details (bool): If True, includes associated document details.
            return_detokenized (bool): If True, returns detokenized text instead of token arrays.
            tokenizer: The tokenizer to use for detokenization (required if return_detokenized is True).

        Raises:
            ValueError: If batch_id is not a non-negative integer or if tokenizer is None when return_detokenized is True.

        Returns:
            List[List[np.ndarray]]: A list of samples, where each sample is a list of token arrays (if return_detokenized is False and return_doc_details is False).
            List[str]: A list of detokenized text samples (if return_detokenized is True and return_doc_details is False).
            List[Tuple[List[np.ndarray], Dict]]: A list of tuples containing token sequences and document details (if return_detokenized is False and return_doc_details is True).
            List[Tuple[str, Dict]]: A list of tuples containing detokenized text and document details (if return_detokenized is True and return_doc_details is True).
        """
        if batch_id < 0:
            raise ValueError("batch_id must be a non-negative integer.")

        if return_detokenized and tokenizer is None:
            raise ValueError("tokenizer must be provided if return_detokenized is True.")

        indices = [i for i in range(batch_id * batch_size, (batch_id + 1) * batch_size)]

        batch_data = []

        for sample_id in indices:
            sample = self.inspect_sample_by_id(
                sample_id=sample_id,
                return_doc_details=return_doc_details,
                return_detokenized=return_detokenized,
                tokenizer=tokenizer
            )
            batch_data.append(sample)

        return batch_data