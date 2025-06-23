from typing import TYPE_CHECKING, Literal, Optional
import numpy as np

if TYPE_CHECKING:
    from ..manager import DatasetManager  # only used for type hints, won't cause import loop

class EditHandler:
    def __init__(self, manager: 'DatasetManager'):
        self.manager = manager

    def inject_and_preview(
        self,
        text: str,
        tokenizer,
        injection_loc: int,
        injection_type: Literal["seq_shuffle", "seq_start"] = "seq_shuffle",
        rng: Optional[np.random.Generator] = None,
        dry_run: bool = False
    ) -> None:
        """
        Injects a dummy sequence into the dataset at a given location and prints before/after samples.

        Parameters:
            text (str): The dummy text to tokenize and inject.
            tokenizer: A HuggingFace-compatible tokenizer with __call__ and decode.
            injection_loc (int): Index of the sample in the training set to modify.
            injection_type (str): Where to inject. Options: 'seq_shuffle' or 'seq_start'.
            rng (np.random.Generator, optional): RNG for reproducibility. If None, uses np.random.default_rng() with seed 1234.
            dry_run (bool): If True, no actual injection is performed.

        Raises:
            ValueError: If injection_loc is negative or injection_type is invalid.
        """
        if not isinstance(injection_loc, int) or injection_loc < 0:
            raise ValueError("injection_loc must be a non-negative integer.")
        if injection_type not in ("seq_shuffle", "seq_start"):
            raise ValueError("injection_type must be 'seq_shuffle' or 'seq_start'.")

        dummy_sample = np.array(
            tokenizer(text, add_special_tokens=True)["input_ids"]
        )
        rng = rng or np.random.default_rng(1234)

        dataset = self.manager.WriteableMMapIndexedDataset

        # Preview original sample
        orig_sample = dataset.get_example_by_id(
            example_loc=injection_loc, return_doc_details=False
        )
        print(f"Training sample {injection_loc}")
        print(f"Sample consists of segments from {len(orig_sample)} documents")

        concat_orig_sample = np.concatenate(orig_sample)
        print(f"Raw sample: {concat_orig_sample}")
        print("---")
        print(f"Decoded sample: {tokenizer.decode(concat_orig_sample)}")
        print("---")

        # Inject
        dataset.inject_example_into_corpus(
            injection_loc=injection_loc,
            injection_data=dummy_sample,
            injection_type=injection_type,
            rng=rng,
            dry_run=dry_run
        )

        # Preview edited sample
        edited_sample = dataset.get_example_by_id(
            example_loc=injection_loc, return_doc_details=False
        )
        print(f"Training sample {injection_loc} after injection")
        concat_edited_sample = np.concatenate(edited_sample)
        print(f"Raw sample: {concat_edited_sample}")
        print("---")
        print(f"Decoded sample: {tokenizer.decode(concat_edited_sample)}")
        print("---")
