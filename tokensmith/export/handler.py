from typing import TYPE_CHECKING, List, Union, Optional, Any, Dict
import json
import csv
import os
from pathlib import Path

if TYPE_CHECKING:
    from ..manager import DatasetManager  # only used for type hints, won't cause import loop

class ExportHandler:
    def __init__(self, manager: 'DatasetManager'):
        self.manager = manager
    
    @staticmethod
    def _normalize_content(content: Any) -> Any:
        """Convert tensors/ndarrays to lists and keep strings/lists JSON-serializable."""
        # Strings are already JSON serializable
        if isinstance(content, str):
            return content
        # Numpy arrays / torch tensors
        if hasattr(content, "tolist"):
            try:
                return content.tolist()
            except Exception:
                pass
        # Python lists/tuples -> list
        if isinstance(content, (list, tuple)):
            return list(content)
        # Numbers or other JSON-serializable primitives
        return content

    @staticmethod
    def _normalize_for_json(obj: Any) -> Any:
        """Recursively convert objects to JSON-serializable types.
        - dict -> dict with normalized values
        - list/tuple/set -> list with normalized items
        - numpy/torch arrays -> tolist()
        - numpy scalars -> item()
        - Path -> str
        Otherwise returned as-is.
        """
        # Primitives
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        # Mapping
        if isinstance(obj, dict):
            return {k: ExportHandler._normalize_for_json(v) for k, v in obj.items()}
        # Sequences/Sets
        if isinstance(obj, (list, tuple, set)):
            return [ExportHandler._normalize_for_json(x) for x in obj]
        # numpy/torch arrays
        if hasattr(obj, "tolist"):
            try:
                return obj.tolist()
            except Exception:
                pass
        # numpy scalar
        if hasattr(obj, "item") and not isinstance(obj, (bytes, bytearray)):
            try:
                return obj.item()
            except Exception:
                pass
        # Path-like
        if isinstance(obj, Path):
            return str(obj)
        return obj

    @staticmethod
    def _stringify_for_csv(value: Any) -> str:
        """Best-effort conversion of values for CSV cell content."""
        v = ExportHandler._normalize_for_json(value)
        # If still a complex type (list/dict), dump as compact JSON
        if isinstance(v, (list, dict)):
            return json.dumps(v, ensure_ascii=False)
        return str(v)

    def export_batches(
        self,
        batch_ids: List[int],
        batch_size: int,
        output_path: str,
        format_type: str = "jsonl",
        return_detokenized: bool = True,
        tokenizer: Optional[Any] = None,
        include_doc_details: bool = False,
        flatten_batches: bool = False
    ) -> None:
        """
        Export specific batches to a file.

        Parameters:
            batch_ids (List[int]): List of batch IDs to export.
            batch_size (int): The size of each batch.
            output_path (str): Path to the output file.
            format_type (str): Format to export ("jsonl" or "csv").
            return_detokenized (bool): If True, exports detokenized text; otherwise exports token arrays.
            tokenizer: The tokenizer to use for detokenization (required if return_detokenized is True).
            include_doc_details (bool): If True, includes document details in the export.
            flatten_batches (bool): If True, flattens all batches into a single list of samples.

        Raises:
            ValueError: If format_type is not supported or tokenizer is None when return_detokenized is True.
        """
        if format_type not in ["jsonl", "csv"]:
            raise ValueError("format_type must be 'jsonl' or 'csv'")

        if return_detokenized and tokenizer is None:
            raise ValueError("tokenizer must be provided if return_detokenized is True")

        # Get batches using the sample handler
        batches = self.manager.sample.get_batches_by_ids(
            batch_ids=batch_ids,
            batch_size=batch_size,
            return_doc_details=include_doc_details,
            return_detokenized=return_detokenized,
            tokenizer=tokenizer
        )

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if format_type == "jsonl":
            self._export_to_jsonl(batches, output_path, flatten_batches, include_doc_details, "batch")
        elif format_type == "csv":
            self._export_to_csv(batches, output_path, flatten_batches, include_doc_details, "batch")

    def export_sequences(
        self,
        sequence_indices: List[int],
        output_path: str,
        format_type: str = "jsonl",
        return_detokenized: bool = True,
        tokenizer: Optional[Any] = None,
        include_doc_details: bool = False
    ) -> None:
        """
        Export specific sequences to a file.

        Parameters:
            sequence_indices (List[int]): List of sequence indices to export.
            output_path (str): Path to the output file.
            format_type (str): Format to export ("jsonl" or "csv").
            return_detokenized (bool): If True, exports detokenized text; otherwise exports token arrays.
            tokenizer: The tokenizer to use for detokenization (required if return_detokenized is True).
            include_doc_details (bool): If True, includes document details in the export.

        Raises:
            ValueError: If format_type is not supported or tokenizer is None when return_detokenized is True.
        """
        if format_type not in ["jsonl", "csv"]:
            raise ValueError("format_type must be 'jsonl' or 'csv'")

        if return_detokenized and tokenizer is None:
            raise ValueError("tokenizer must be provided if return_detokenized is True")

        # Get samples using the sample handler
        samples = self.manager.sample.get_samples_by_indices(
            indices=sequence_indices,
            return_doc_details=include_doc_details,
            return_detokenized=return_detokenized,
            tokenizer=tokenizer
        )

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if format_type == "jsonl":
            self._export_to_jsonl([samples], output_path, True, include_doc_details, "sequence")
        elif format_type == "csv":
            self._export_to_csv([samples], output_path, True, include_doc_details, "sequence")

    def export_entire_dataset(
        self,
        output_path: str,
        format_type: str = "jsonl",
        return_detokenized: bool = True,
        tokenizer: Optional[Any] = None,
        include_doc_details: bool = False,
        chunk_size: int = 1000
    ) -> None:
        """
        Export the entire dataset to a file.

        Parameters:
            output_path (str): Path to the output file.
            format_type (str): Format to export ("jsonl" or "csv").
            return_detokenized (bool): If True, exports detokenized text; otherwise exports token arrays.
            tokenizer: The tokenizer to use for detokenization (required if return_detokenized is True).
            include_doc_details (bool): If True, includes document details in the export.
            chunk_size (int): Number of samples to process at a time to manage memory usage.

        Raises:
            ValueError: If format_type is not supported or tokenizer is None when return_detokenized is True.
        """
        if format_type not in ["jsonl", "csv"]:
            raise ValueError("format_type must be 'jsonl' or 'csv'")

        if return_detokenized and tokenizer is None:
            raise ValueError("tokenizer must be provided if return_detokenized is True")

        # Get total number of samples
        total_samples = len(self.manager.WriteableMMapIndexedDataset.batch_info.shuffle_idx)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Export in chunks to manage memory
        if format_type == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for start_idx in range(0, total_samples, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_samples)
                    chunk_indices = list(range(start_idx, end_idx))
                    
                    chunk_samples = self.manager.sample.get_samples_by_indices(
                        indices=chunk_indices,
                        return_doc_details=include_doc_details,
                        return_detokenized=return_detokenized,
                        tokenizer=tokenizer
                    )
                    
                    self._write_chunk_to_jsonl(chunk_samples, f, include_doc_details, start_idx)

        elif format_type == "csv":
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = None
                for start_idx in range(0, total_samples, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_samples)
                    chunk_indices = list(range(start_idx, end_idx))
                    
                    chunk_samples = self.manager.sample.get_samples_by_indices(
                        indices=chunk_indices,
                        return_doc_details=include_doc_details,
                        return_detokenized=return_detokenized,
                        tokenizer=tokenizer
                    )
                    
                    writer = self._write_chunk_to_csv(chunk_samples, f, writer, include_doc_details, start_idx)

    def export_sequence_range(
        self,
        start_idx: int,
        end_idx: int,
        output_path: str,
        format_type: str = "jsonl",
        return_detokenized: bool = True,
        tokenizer: Optional[Any] = None,
        include_doc_details: bool = False
    ) -> None:
        """
        Export a range of sequences to a file.

        Parameters:
            start_idx (int): Starting sequence index (inclusive).
            end_idx (int): Ending sequence index (exclusive).
            output_path (str): Path to the output file.
            format_type (str): Format to export ("jsonl" or "csv").
            return_detokenized (bool): If True, exports detokenized text; otherwise exports token arrays.
            tokenizer: The tokenizer to use for detokenization (required if return_detokenized is True).
            include_doc_details (bool): If True, includes document details in the export.

        Raises:
            ValueError: If format_type is not supported, tokenizer is None when return_detokenized is True,
                       or if start_idx >= end_idx or indices are negative.
        """
        if format_type not in ["jsonl", "csv"]:
            raise ValueError("format_type must be 'jsonl' or 'csv'")

        if return_detokenized and tokenizer is None:
            raise ValueError("tokenizer must be provided if return_detokenized is True")

        if not isinstance(start_idx, int) or not isinstance(end_idx, int):
            raise ValueError("start_idx and end_idx must be integers")

        if start_idx < 0 or end_idx < 0:
            raise ValueError("start_idx and end_idx must be non-negative")

        if start_idx >= end_idx:
            raise ValueError("start_idx must be less than end_idx")

        # Generate sequence indices for the range
        sequence_indices = list(range(start_idx, end_idx))

        # Use the existing export_sequences method
        self.export_sequences(
            sequence_indices=sequence_indices,
            output_path=output_path,
            format_type=format_type,
            return_detokenized=return_detokenized,
            tokenizer=tokenizer,
            include_doc_details=include_doc_details
        )

    def export_batch_range(
        self,
        start_batch: int,
        end_batch: int,
        batch_size: int,
        output_path: str,
        format_type: str = "jsonl",
        return_detokenized: bool = True,
        tokenizer: Optional[Any] = None,
        include_doc_details: bool = False,
        flatten_batches: bool = False
    ) -> None:
        """
        Export a range of batches to a file.

        Parameters:
            start_batch (int): Starting batch ID (inclusive).
            end_batch (int): Ending batch ID (exclusive).
            batch_size (int): The size of each batch.
            output_path (str): Path to the output file.
            format_type (str): Format to export ("jsonl" or "csv").
            return_detokenized (bool): If True, exports detokenized text; otherwise exports token arrays.
            tokenizer: The tokenizer to use for detokenization (required if return_detokenized is True).
            include_doc_details (bool): If True, includes document details in the export.
            flatten_batches (bool): If True, flattens all batches into a single list of samples.

        Raises:
            ValueError: If format_type is not supported, tokenizer is None when return_detokenized is True,
                       or if start_batch >= end_batch or batch IDs are negative.
        """
        if format_type not in ["jsonl", "csv"]:
            raise ValueError("format_type must be 'jsonl' or 'csv'")

        if return_detokenized and tokenizer is None:
            raise ValueError("tokenizer must be provided if return_detokenized is True")

        if not isinstance(start_batch, int) or not isinstance(end_batch, int):
            raise ValueError("start_batch and end_batch must be integers")

        if start_batch < 0 or end_batch < 0:
            raise ValueError("start_batch and end_batch must be non-negative")

        if start_batch >= end_batch:
            raise ValueError("start_batch must be less than end_batch")

        # Generate batch IDs for the range
        batch_ids = list(range(start_batch, end_batch))

        # Use the existing export_batches method
        self.export_batches(
            batch_ids=batch_ids,
            batch_size=batch_size,
            output_path=output_path,
            format_type=format_type,
            return_detokenized=return_detokenized,
            tokenizer=tokenizer,
            include_doc_details=include_doc_details,
            flatten_batches=flatten_batches
        )

    def export_dataset_range(
        self,
        start_idx: int,
        end_idx: int,
        output_path: str,
        format_type: str = "jsonl",
        return_detokenized: bool = True,
        tokenizer: Optional[Any] = None,
        include_doc_details: bool = False,
        chunk_size: int = 1000
    ) -> None:
        """
        Export a range of the dataset to a file with memory-efficient chunking.

        Parameters:
            start_idx (int): Starting sequence index (inclusive).
            end_idx (int): Ending sequence index (exclusive).
            output_path (str): Path to the output file.
            format_type (str): Format to export ("jsonl" or "csv").
            return_detokenized (bool): If True, exports detokenized text; otherwise exports token arrays.
            tokenizer: The tokenizer to use for detokenization (required if return_detokenized is True).
            include_doc_details (bool): If True, includes document details in the export.
            chunk_size (int): Number of samples to process at a time to manage memory usage.

        Raises:
            ValueError: If format_type is not supported, tokenizer is None when return_detokenized is True,
                       or if start_idx >= end_idx or indices are negative.
        """
        if format_type not in ["jsonl", "csv"]:
            raise ValueError("format_type must be 'jsonl' or 'csv'")

        if return_detokenized and tokenizer is None:
            raise ValueError("tokenizer must be provided if return_detokenized is True")

        if not isinstance(start_idx, int) or not isinstance(end_idx, int):
            raise ValueError("start_idx and end_idx must be integers")

        if start_idx < 0 or end_idx < 0:
            raise ValueError("start_idx and end_idx must be non-negative")

        if start_idx >= end_idx:
            raise ValueError("start_idx must be less than end_idx")

        # Get total number of samples to validate range
        # total_samples = len(self.manager.WriteableMMapIndexedDataset.batch_info.shuffle_idx)
        # if end_idx > total_samples:
        #     raise ValueError(f"end_idx ({end_idx}) exceeds dataset size ({total_samples})")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Export in chunks to manage memory
        if format_type == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                current_idx = start_idx
                while current_idx < end_idx:
                    chunk_end = min(current_idx + chunk_size, end_idx)
                    chunk_indices = list(range(current_idx, chunk_end))
                    
                    chunk_samples = self.manager.sample.get_samples_by_indices(
                        indices=chunk_indices,
                        return_doc_details=include_doc_details,
                        return_detokenized=return_detokenized,
                        tokenizer=tokenizer
                    )
                    
                    self._write_chunk_to_jsonl(chunk_samples, f, include_doc_details, current_idx)
                    current_idx = chunk_end

        elif format_type == "csv":
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = None
                current_idx = start_idx
                while current_idx < end_idx:
                    chunk_end = min(current_idx + chunk_size, end_idx)
                    chunk_indices = list(range(current_idx, chunk_end))
                    
                    chunk_samples = self.manager.sample.get_samples_by_indices(
                        indices=chunk_indices,
                        return_doc_details=include_doc_details,
                        return_detokenized=return_detokenized,
                        tokenizer=tokenizer
                    )
                    
                    writer = self._write_chunk_to_csv(chunk_samples, f, writer, include_doc_details, current_idx)
                    current_idx = chunk_end

    def _export_to_jsonl(
        self,
        data: List[Any],
        output_path: str,
        flatten: bool,
        include_doc_details: bool,
        export_type: str
    ) -> None:
        """Export data to JSONL format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            # If exporting batches and not flattening, write one JSON object per batch
            if not flatten and export_type == "batch":
                for batch_idx, batch in enumerate(data):
                    if include_doc_details:
                        samples = []
                        for item in batch:
                            content, doc_details = item
                            samples.append({
                                "content": self._normalize_content(content),
                                "doc_details": self._normalize_for_json(doc_details)
                            })
                    else:
                        samples = [self._normalize_content(item) for item in batch]

                    record = {
                        "batch_id": batch_idx,
                        "samples": samples
                    }
                    json.dump(record, f, ensure_ascii=False)
                    f.write('\n')
                return

            # Default behavior: write a flat list of samples
            if flatten:
                # Flatten all batches/sequences into a single list
                flattened_data = []
                for batch in data:
                    flattened_data.extend(batch)
                data = flattened_data

            self._write_chunk_to_jsonl(data, f, include_doc_details, 0)

    def _export_to_csv(
        self,
        data: List[Any],
        output_path: str,
        flatten: bool,
        include_doc_details: bool,
        export_type: str
    ) -> None:
        """Export data to CSV format."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if flatten:
                # Flatten all batches/sequences into a single list
                flattened_data = []
                for batch in data:
                    flattened_data.extend(batch)
                data = flattened_data

            self._write_chunk_to_csv(data, f, None, include_doc_details, 0)

    def _write_chunk_to_jsonl(
        self,
        chunk_data: List[Any],
        file_handle,
        include_doc_details: bool,
        start_idx: int
    ) -> None:
        """Write a chunk of data to JSONL format."""
        for i, sample in enumerate(chunk_data):
            if include_doc_details:
                content, doc_details = sample
                record = {
                    "index": start_idx + i,
                    "content": self._normalize_content(content),
                    "doc_details": self._normalize_for_json(doc_details)
                }
            else:
                record = {
                    "index": start_idx + i,
                    "content": self._normalize_content(sample)
                }
            
            json.dump(record, file_handle, ensure_ascii=False)
            file_handle.write('\n')

    def _write_chunk_to_csv(
        self,
        chunk_data: List[Any],
        file_handle,
        writer,
        include_doc_details: bool,
        start_idx: int
    ):
        """Write a chunk of data to CSV format."""
        if writer is None:
            # Determine fieldnames based on the first sample
            if include_doc_details and chunk_data:
                sample_content, sample_doc_details = chunk_data[0]
                fieldnames = ["index", "content"]
                if sample_doc_details:
                    fieldnames.extend(sample_doc_details.keys())
            else:
                fieldnames = ["index", "content"]
            
            writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
            writer.writeheader()

        for i, sample in enumerate(chunk_data):
            if include_doc_details:
                content, doc_details = sample
                row = {
                    "index": start_idx + i,
                    "content": ExportHandler._stringify_for_csv(content)
                }
                if doc_details:
                    row.update({k: ExportHandler._stringify_for_csv(v) for k, v in doc_details.items()})
            else:
                row = {
                    "index": start_idx + i,
                    "content": ExportHandler._stringify_for_csv(sample)
                }
            
            writer.writerow(row)
        
        return writer