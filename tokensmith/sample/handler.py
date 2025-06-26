from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ..manager import DatasetManager  # only used for type hints, won't cause import loop

class SampleHandler:
    def __init__(self, manager: 'DatasetManager'):
        self.manager = manager

    # Write a method that accepts a list of indexes to sample and returns a list of samples.
    def sample_by_indexes(self, indexes: List, return_doc_details: bool = False) -> List:
        """Returns a list of samples by their indexes."""
        if not isinstance(indexes, list):
            raise ValueError("indexes must be a list.")
        if not all(isinstance(i, int) for i in indexes):
            raise ValueError("All elements in indexes must be integers.")
        if not all(i >= 0 for i in indexes):
            raise ValueError("All elements in indexes must be non-negative integers.")

        if return_doc_details:
            samples = [
                self.manager.WriteableMMapIndexedDataset.get_example_by_id(
                    example_loc=index,
                    return_doc_details=return_doc_details
                ) for index in indexes
            ]
        else:
            samples = [
                self.manager.WriteableMMapIndexedDataset.get_example_by_id(
                    example_loc=index,
                    return_doc_details=return_doc_details
                ) for index in indexes
            ]
        return samples