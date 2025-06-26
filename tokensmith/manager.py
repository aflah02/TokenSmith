from typing import Dict, Any, Optional
from tokensmith.edit import EditHandler
from tokensmith.inspect import InspectHandler
from tokensmith.search import SearchHandler
from tokensmith.sample import SampleHandler
from tokensmith.export import ExportHandler
from tokensmith.utils import WriteableMMapIndexedDataset
from typing import Optional

class DatasetManager:
    def __init__(self):
        # Edit, Inspect, Sample, and Export handlers are initialized to None and will be set up when setup_edit_inspect_sample_export is called
        self.edit: Optional[EditHandler] = None
        self.inspect: Optional[InspectHandler] = None
        self.sample: Optional[SampleHandler] = None
        self.export: Optional[ExportHandler] = None

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
            print(SearchHandler)
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

    def setup_edit_inspect_sample_export(self, dataset_prefix: str, batch_info_save_prefix: str,
                                         train_iters: int, train_batch_size: int, train_seq_len: int, seed: int, splits_string: str = '969,30,1',
                                         packing_impl: str = 'packed',
                                         allow_chopped: bool = True,
                                         add_extra_token_to_seq: int = 1):
        """
        Initializes the EditHandler, InspectHandler, SampleHandler, and ExportHandler.
        This method is called to set up the handlers with the provided bin file path.

        Parameters:
            dataset_prefix (str): Prefix for the dataset files. This is used to locate the {dataset_prefix}.bin and {dataset_prefix}.idx files.
            batch_info_save_prefix (str): Prefix for the batch information files. This is used to locate the doc/sample/shuffle indexes with the given prefix/save path if the files are not found.
            train_iters (int): Number of training iterations for simulated training.
            train_batch_size (int): Size of each training batch for simulated training.
            train_seq_len (int): Length of the training sequences.
            seed (int): Random seed for simulated training.
            splits_string (str): Comma-separated string of train/val/test splits. (defaults to '969,30,1' which means 96.9% train, 3% val, and 0.1% test).
            packing_impl (str): Implementation for packing sequences. One of 'packed', 'pack_until_overflow', 'unpacked'. (defaults to 'packed').
            allow_chopped (bool): WARNING: if your packing impl is packed, this is ignored. Allow chopped samples in the dataset. E.g if your sequence length is 1024 and you have a sample of length 1026, it will be chopped to 1024 (defaults to True).
            add_extra_token_to_seq (int): Number of extra tokens to add to each sequence (defaults to 1 to account for causal language modeling).

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
            batch_info_save_prefix=batch_info_save_prefix,
            train_iters=train_iters,
            train_batch_size=train_batch_size,
            seed=seed,
            splits_string=splits_string,
            packing_impl=packing_impl,
            allow_chopped=allow_chopped,
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
        

    