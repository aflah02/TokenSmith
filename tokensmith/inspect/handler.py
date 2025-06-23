from typing import TYPE_CHECKING, Union, List, Dict, Tuple
import numpy as np

if TYPE_CHECKING:
    from ..manager import DatasetManager  # only used for type hints, won't cause import loop

class InspectHandler:
    def __init__(self, manager: 'DatasetManager'):
        self.manager = manager

    def inspect_sample_by_id(
        self,
        sample_id: int,
        return_doc_details: bool = False
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], Dict]]:
        """
        Returns a sample by its ID, optionally with document details.

        Parameters:
            sample_id (int): The index of the sample to retrieve.
            return_doc_details (bool): If True, includes associated document details.

        Raises:
            ValueError: If sample_id is not a non-negative integer.

        Returns:
            list of np.ndarray: A list of numpy arrays representing the sample sequence (different arrays showcase the splits from different documents).
            tuple (list of np.ndarray, dict): If return_doc_details is True, also returns a dictionary with document details.
        """
        if sample_id < 0:
            raise ValueError("sample_id must be a non-negative integer.")

        return self.manager.WriteableMMapIndexedDataset.get_example_by_id(
            example_loc=sample_id,
            return_doc_details=return_doc_details
        )