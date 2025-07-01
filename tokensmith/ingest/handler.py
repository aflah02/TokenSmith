from typing import TYPE_CHECKING, Optional, Dict
import os
import subprocess
import logging
import pandas as pd
import json
from pathlib import Path

if TYPE_CHECKING:
    from ..manager import DatasetManager

logger = logging.getLogger(__name__)

class IngestHandler:
    def __init__(self, manager: 'DatasetManager'):
        self.manager = manager

    def ingest_from_jsonl(
        self,
        input_jsonl_path: str,
        output_prefix: str,
        vocab_path: str,
        neox_dir: str,
        workers: int,
        append_eod: bool,
        dataset_impl: str,
        tokenizer_type: str,
        log_file: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Ingest and tokenize a JSONL file using GPT-NeoX preprocessing pipeline.
        
        Parameters:
            input_jsonl_path (str): Path to the input JSONL file.
            output_prefix (str): Prefix for the output tokenized files.
            vocab_path (str): Path to the vocab file.
            neox_dir (str): Path to the GPT-NeoX directory.
            workers (int): Number of workers for tokenization.
            append_eod (bool): Whether to append end-of-document tokens.
            dataset_impl (str): Dataset implementation type.
            tokenizer_type (str): Type of tokenizer to use.
            log_file (Optional[str]): Path to save tokenization logs.
            
        Returns:
            Dict[str, str]: Dictionary containing paths to generated files.
            
        Raises:
            FileNotFoundError: If input files or directories don't exist.
            subprocess.CalledProcessError: If tokenization process fails.
        """
        # Validate inputs
        if not os.path.exists(input_jsonl_path):
            raise FileNotFoundError(f"Input JSONL file not found: {input_jsonl_path}")
        
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Tokenizer file not found: {vocab_path}")
            
        if not os.path.exists(neox_dir):
            raise FileNotFoundError(f"GPT-NeoX directory not found: {neox_dir}")
            
        preprocess_script = os.path.join(neox_dir, "tools/datasets/preprocess_data.py")
        if not os.path.exists(preprocess_script):
            raise FileNotFoundError(f"Preprocessing script not found: {preprocess_script}")

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_prefix)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Set up log file
        if log_file is None:
            log_file = f"{output_prefix}_tokenize.log"

        logger.info(f"Starting tokenization of {input_jsonl_path}")
        logger.info(f"Output prefix: {output_prefix}")
        logger.info(f"Using vocab file: {vocab_path}")

        # Build the command exactly like tokenize.sh
        cmd = [
            "python", preprocess_script,
            "--input", input_jsonl_path,
            "--output-prefix", output_prefix,
            "--vocab", vocab_path,
            "--dataset-impl", dataset_impl,
            "--tokenizer-type", tokenizer_type,
            "--workers", str(workers)
        ]
        
        if append_eod:
            cmd.append("--append-eod")

        try:
            # Run the tokenization process with tee-like logging
            with open(log_file, 'w') as log_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Stream output to both log file and logger
                for line in process.stdout:
                    log_f.write(line)
                    log_f.flush()
                    logger.info(line.strip())
                
                process.wait()
                
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd)
            
            logger.info(f"Tokenization completed successfully. Log file: {log_file}")
            
            # Verify output files were created
            expected_files = {
                "bin_file": f"{output_prefix}_text_document.bin",
                "idx_file": f"{output_prefix}_text_document.idx",
                "log_file": log_file
            }
            
            missing_files = []
            for file_type, file_path in expected_files.items():
                if not os.path.exists(file_path):
                    missing_files.append(f"{file_type}: {file_path}")
            
            if missing_files:
                raise FileNotFoundError(f"Expected output files not found: {missing_files}")
            
            return expected_files
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Tokenization failed with return code {e.returncode}")
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_content = f.read()
                logger.error(f"Tokenization log:\n{log_content}")
            raise

    def ingest_from_csv(
        self,
        input_csv_path: str,
        text_column: str,
        output_prefix: str,
        vocab_path: str,
        neox_dir: str,
        workers: int,
        append_eod: bool,
        dataset_impl: str,
        tokenizer_type: str,
        log_file: Optional[str] = None,
        chunk_size: int = 10000,
        cleanup_temp: bool = True
    ) -> Dict[str, str]:
        """
        Fast import and tokenize a CSV file by converting to JSONL first.
        
        Parameters:
            input_csv_path (str): Path to the input CSV file.
            text_column (str): Name of the column containing text data.
            output_prefix (str): Prefix for the output tokenized files.
            vocab_path (str): Path to the vocab file.
            neox_dir (str): Path to the GPT-NeoX directory.
            workers (int): Number of workers for tokenization.
            append_eod (bool): Whether to append end-of-document tokens.
            dataset_impl (str): Dataset implementation type.
            tokenizer_type (str): Type of tokenizer to use.
            log_file (Optional[str]): Path to save tokenization logs.
            chunk_size (int): Number of rows to process at once for memory efficiency.
            cleanup_temp (bool): Whether to clean up temporary JSONL file.
            
        Returns:
            Dict[str, str]: Dictionary containing paths to generated files.
            
        Raises:
            FileNotFoundError: If input file doesn't exist.
            ValueError: If text_column doesn't exist in CSV.
        """
        if not os.path.exists(input_csv_path):
            raise FileNotFoundError(f"Input CSV file not found: {input_csv_path}")

        # Create temporary JSONL file
        temp_jsonl_path = f"{output_prefix}_temp.jsonl"
        
        logger.info(f"Converting CSV to JSONL: {input_csv_path} -> {temp_jsonl_path}")
        
        try:
            # Read CSV in chunks for memory efficiency
            total_rows = 0
            with open(temp_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
                for chunk_df in pd.read_csv(input_csv_path, chunksize=chunk_size):
                    # Validate that text column exists
                    if text_column not in chunk_df.columns:
                        raise ValueError(f"Column '{text_column}' not found in CSV. Available columns: {list(chunk_df.columns)}")
                    
                    # Convert each row to JSONL format
                    for _, row in chunk_df.iterrows():
                        text_content = str(row[text_column]).strip()
                        if text_content:  # Skip empty rows
                            json_record = {"text": text_content}
                            jsonl_file.write(json.dumps(json_record, ensure_ascii=False) + '\n')
                            total_rows += 1
            
            logger.info(f"Converted {total_rows} rows from CSV to JSONL")
            
            # Now use the existing ingest_from_jsonl method
            result = self.ingest_from_jsonl(
                input_jsonl_path=temp_jsonl_path,
                output_prefix=output_prefix,
                vocab_path=vocab_path,
                neox_dir=neox_dir,
                workers=workers,
                append_eod=append_eod,
                dataset_impl=dataset_impl,
                tokenizer_type=tokenizer_type,
                log_file=log_file
            )
            
            # Add info about the conversion
            result["source_csv"] = input_csv_path
            result["text_column"] = text_column
            result["total_rows"] = total_rows
            result["temp_jsonl"] = temp_jsonl_path
            
            return result
            
        except Exception as e:
            logger.error(f"Error converting CSV to JSONL: {e}")
            raise
        finally:
            # Clean up temporary file if requested
            if cleanup_temp and os.path.exists(temp_jsonl_path):
                os.remove(temp_jsonl_path)
                logger.info(f"Cleaned up temporary file: {temp_jsonl_path}")