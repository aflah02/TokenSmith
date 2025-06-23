from typing import Dict, Any, Optional
from tokensmith.edit import EditHandler
from tokensmith.inspect import InspectHandler
from tokensmith.search import SearchHandler
from tokensmith.sample import SampleHandler
from tokensmith.export import ExportHandler
from utils import WriteableMMapIndexedDataset
from typing import Optional

class DatasetManager:
    def __init__(self):
        # Edit, Inspect, Sample, and Export handlers are initialized to None and will be set up when setup_edit_inspect_sample_export is called
        self.edit = Optional[EditHandler] = None
        self.inspect = Optional[InspectHandler] = None
        self.sample = Optional[SampleHandler] = None
        self.export = Optional[ExportHandler] = None

        # SearchHandler will be initialized when setup_search is called
        self.search: Optional[SearchHandler] = None

    def setup_search(self, bin_file_path: str, search_index_save_path: str, vocab: int, verbose: bool = False, reuse: bool = True):
        """
        Initializes the SearchHandler by building or loading the index.
        Should be called explicitly if search functionality is required.
        Not done automatically to avoid unnecessary overhead.

        Parameters:
            bin_save_path (str): Path to the binary file containing the dataset.
            search_index_save_path (str): Path to save the search index.
            vocab (Dict[str, int]): Vocabulary mapping words to their indices.
            verbose (bool): If True, enables verbose output during index building.
            reuse (bool): If True, reuses the existing index if available.
        
        Raises:
            ValueError: 
                - If SearchHandler is already initialized.
                - If vocab is not 2**16 or 2**32.
                - If reuse is True but the index path does not exist.

        Returns:
            None
        """
        if self.search is None:
            self.search = SearchHandler(
                bin_file_path=bin_file_path,
                index_save_path=search_index_save_path,
                vocab=vocab,
                verbose=verbose,
                reuse=reuse
            )
        else:
            raise ValueError(
                "SearchHandler already initialized. "
                "Create a new DatasetManager instance or reset `search` manually."
            )

    def setup_edit_inspect_sample_export(self, dataset_prefix: str, batch_info_prefix: str, train_seq_len: int, add_extra_token_to_seq: int = 1):
        """
        Initializes the EditHandler, InspectHandler, SampleHandler, and ExportHandler.
        This method is called to set up the handlers with the provided bin file path.

        Parameters:
            dataset_prefix (str): Prefix for the dataset files. This is used to locate the {dataset_prefix}.bin and {dataset_prefix}.idx files.
            batch_info_prefix (str): Prefix for the batch information files. This is used to locate the doc/sample/shuffle indexes with the given prefix
            train_seq_len (int): Length of the training sequences.
            add_extra_token_to_seq (int): Number of extra tokens to add to each sequence (default to 1 to account adding EOS token)

        Raises:
            ValueError: 
                - If any of the handlers are already initialized.
                - If train_seq_len is not a positive integer.
                - If add_extra_token_to_seq is not a non-negative integer.

        Returns:
            None
        """

        self.WriteableMMapIndexedDataset = WriteableMMapIndexedDataset(
            dataset_prefix=dataset_prefix,
            batch_info_prefix=batch_info_prefix,
            train_seq_len=train_seq_len,
            add_extra_token_to_seq=add_extra_token_to_seq
        )

        if self.edit is None:
            self.edit = EditHandler(self)
        else:
            raise ValueError("EditHandler already initialized. Create a new DatasetManager instance or reset `edit` manually.")

        if self.inspect is None:
            self.inspect = InspectHandler(self)
        else:
            raise ValueError("InspectHandler already initialized. Create a new DatasetManager instance or reset `inspect` manually.")

        if self.sample is None:
            self.sample = SampleHandler(self)
        else:
            raise ValueError("SampleHandler already initialized. Create a new DatasetManager instance or reset `sample` manually.")

        if self.export is None:
            self.export = ExportHandler(self)
        else:
            raise ValueError("ExportHandler already initialized. Create a new DatasetManager instance or reset `export` manually.")
        

    