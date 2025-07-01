
from typing import TYPE_CHECKING, Union, List, Dict, Tuple, Optional, Any, Literal
import numpy as np
import warnings
import logging
from ..utils import generate_training_sample

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..manager import DatasetManager  # only used for type hints, won't cause import loop

class EditHandler:
    def __init__(self, manager: 'DatasetManager'):
        self.manager = manager

    def inject_and_preview(
        self,
        text: str,
        tokenizer: Optional[Any],
        injection_loc: int,
        injection_type: Literal["seq_shuffle", "seq_start"] = "seq_shuffle",
        rng: Optional[np.random.Generator] = None,
        add_eos_token: bool = True,
        dry_run: bool = True,
        return_details: bool = False
    ) -> Union[None, Dict[str, Any]]:
        """
        Injects a dummy sequence into the dataset at a given location and prints before/after samples.

        Parameters:
            text (str): The dummy text to tokenize and inject.
            tokenizer: A HuggingFace-compatible tokenizer with __call__ and decode.
            injection_loc (int): Index of the sample in the training set to modify.
            injection_type (str): Where to inject. Options: 'seq_shuffle' or 'seq_start'.
            rng (np.random.Generator, optional): RNG for reproducibility. If None, uses np.random.default_rng() with seed 1234.
            add_eos_token (bool): Whether to add EOS token to the injected text.
            dry_run (bool): If True, no actual injection is performed.
            return_details (bool): If True, returns structured data instead of just printing.

        Raises:
            ValueError: If injection_loc is negative, injection_type is invalid, or tokenizer is None.

        Returns:
            None: If return_details is False (default behavior with printing).
            Dict[str, Any]: If return_details is True, returns structured data with original and modified samples.
        """
        # Input validation
        if not isinstance(text, str):
            raise ValueError("text must be a string.")
        if tokenizer is None:
            raise ValueError("tokenizer must be provided.")
        if not isinstance(injection_loc, int) or injection_loc < 0:
            raise ValueError("injection_loc must be a non-negative integer.")
        if injection_type not in ("seq_shuffle", "seq_start"):
            raise ValueError("injection_type must be 'seq_shuffle' or 'seq_start'.")

        # Tokenize the input text
        try:
            dummy_sample = np.array(tokenizer(text)["input_ids"])
        except Exception as e:
            raise ValueError(f"Failed to tokenize input text: {e}")

        # Add EOS token if requested
        if add_eos_token and hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            if len(dummy_sample) > 0 and dummy_sample[-1] == tokenizer.eos_token_id:
                warnings.warn("The injected sample already contains the EOS token.")
            else:
                dummy_sample = np.append(dummy_sample, tokenizer.eos_token_id)
        elif add_eos_token:
            warnings.warn("Tokenizer does not have an EOS token, skipping EOS token addition.")
        
        if not return_details:
            print(f"Dummy sample: {dummy_sample}")

        rng = rng or np.random.default_rng(1234)
        dataset = self.manager.WriteableMMapIndexedDataset

        # Get original sample
        try:
            orig_sample = dataset.get_example_by_id(
                example_loc=injection_loc, return_doc_details=False
            )
        except Exception as e:
            raise ValueError(f"Failed to retrieve sample at location {injection_loc}: {e}")

        concat_orig_sample = np.concatenate(orig_sample)
        orig_decoded = tokenizer.decode(concat_orig_sample) if hasattr(tokenizer, 'decode') else str(concat_orig_sample)

        if not return_details:
            print(f"Training sample {injection_loc}")
            print(f"Sample consists of segments from {len(orig_sample)} documents")
            print(f"Raw sample: {concat_orig_sample}")
            print("---")
            print(f"Decoded sample: {orig_decoded}")
            print("---")

        # Perform injection
        try:
            injection_details = dataset.inject_example_into_corpus(
                injection_loc=injection_loc,
                injection_data=dummy_sample,
                injection_type=injection_type,
                rng=rng,
                dry_run=dry_run
            )
        except Exception as e:
            raise ValueError(f"Failed to inject sample: {e}")

        # Get modified sample
        try:
            edited_sample = dataset.get_example_by_id(
                example_loc=injection_loc, return_doc_details=False
            )
        except Exception as e:
            raise ValueError(f"Failed to retrieve modified sample: {e}")

        concat_edited_sample = np.concatenate(edited_sample)
        edited_decoded = tokenizer.decode(concat_edited_sample) if hasattr(tokenizer, 'decode') else str(concat_edited_sample)

        if not return_details:
            print(f"Training sample {injection_loc} after injection")
            print(f"Raw sample: {concat_edited_sample}")
            print("---")
            print(f"Decoded sample: {edited_decoded}")
            print("---")
        else:
            return {
                "injection_location": injection_loc,
                "injection_type": injection_type,
                "dry_run": dry_run,
                "injected_text": text,
                "injected_tokens": dummy_sample.tolist(),
                "original_sample": {
                    "raw_tokens": concat_orig_sample.tolist(),
                    "decoded_text": orig_decoded,
                    "num_documents": len(orig_sample)
                },
                "modified_sample": {
                    "raw_tokens": concat_edited_sample.tolist(),
                    "decoded_text": edited_decoded,
                    "num_documents": len(edited_sample)
                },
                "injection_details": injection_details
            }

    def inject_multiple_samples(
        self,
        injections: List[Dict[str, Any]],
        tokenizer: Optional[Any],
        rng: Optional[np.random.Generator] = None,
        add_eos_token: bool = True,
        dry_run: bool = True,
        return_details: bool = False
    ) -> Union[None, List[Dict[str, Any]]]:
        """
        Inject multiple samples into the dataset in batch.

        Parameters:
            injections (List[Dict]): List of injection specifications, each containing:
                - text (str): Text to inject
                - injection_loc (int): Location to inject
                - injection_type (str, optional): Type of injection, defaults to "seq_shuffle"
            tokenizer: A HuggingFace-compatible tokenizer.
            rng (np.random.Generator, optional): RNG for reproducibility.
            add_eos_token (bool): Whether to add EOS token to injected text.
            dry_run (bool): If True, no actual injection is performed.
            return_details (bool): If True, returns structured data for all injections.

        Raises:
            ValueError: If injections list is invalid or any injection specification is invalid.

        Returns:
            None: If return_details is False.
            List[Dict[str, Any]]: If return_details is True, returns list of injection results.
        """
        if not isinstance(injections, list):
            raise ValueError("injections must be a list.")
        if not injections:
            raise ValueError("injections list cannot be empty.")
        if tokenizer is None:
            raise ValueError("tokenizer must be provided.")

        results = []
        for i, injection in enumerate(injections):
            if not isinstance(injection, dict):
                raise ValueError(f"Injection {i} must be a dictionary.")
            
            if "text" not in injection or "injection_loc" not in injection:
                raise ValueError(f"Injection {i} must contain 'text' and 'injection_loc' keys.")
            
            injection_type = injection.get("injection_type", "seq_shuffle")
            
            try:
                result = self.inject_and_preview(
                    text=injection["text"],
                    tokenizer=tokenizer,
                    injection_loc=injection["injection_loc"],
                    injection_type=injection_type,
                    rng=rng,
                    add_eos_token=add_eos_token,
                    dry_run=dry_run,
                    return_details=True
                )
                
                if return_details:
                    results.append(result)
                else:
                    print(f"=== Injection {i + 1}/{len(injections)} ===")
                    
            except Exception as e:
                error_msg = f"Failed to process injection {i}: {e}"
                logger.error(error_msg)
                if return_details:
                    results.append({"error": error_msg, "injection_index": i})
                else:
                    print(f"ERROR: {error_msg}")

        if return_details:
            return results

    def preview_sample(
        self,
        sample_id: int,
        return_doc_details: bool = False,
        return_detokenized: bool = True,
        tokenizer: Optional[Any] = None,
    ) -> Union[List[np.ndarray], str, Tuple[List[np.ndarray], Dict], Tuple[str, Dict]]:
        """
        Preview a sample by its ID without modification, similar to inspect functionality.

        Parameters:
            sample_id (int): The index of the sample to preview.
            return_doc_details (bool): If True, includes associated document details.
            return_detokenized (bool): If True, returns detokenized text instead of token arrays.
            tokenizer: The tokenizer to use for detokenization (required if return_detokenized is True).

        Raises:
            ValueError: If sample_id is not a non-negative integer or if tokenizer is None when return_detokenized is True.

        Returns:
            Similar to InspectHandler.inspect_sample_by_id
        """
        if not isinstance(sample_id, int) or sample_id < 0:
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

    def validate_injection_location(self, injection_loc: int) -> bool:
        """
        Validate if an injection location is valid for the dataset.

        Parameters:
            injection_loc (int): The injection location to validate.

        Returns:
            bool: True if the location is valid, False otherwise.
        """
        if not isinstance(injection_loc, int) or injection_loc < 0:
            return False
        
        try:
            return injection_loc < self.WriteableMMapIndexedDataset.num_samples
        except Exception:
            return False

