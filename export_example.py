#!/usr/bin/env python3
"""
Example usage of the Export Handler functionality.
This demonstrates how to export batches, sequences, and entire datasets.
"""

from tokensmith import DatasetManager
from transformers import AutoTokenizer

def main():
    # Initialize the DatasetManager
    manager = DatasetManager()
    
    # Setup the dataset for edit, inspect, sample, and export operations
    manager.setup_edit_inspect_sample_export(
        dataset_prefix="artifacts/data_tokenized_text_document",
        batch_info_save_prefix="artifacts/batch_info_train_indexmap_1600ns_2048sl_42s_packedpi_ac",
        train_iters=1600,
        train_batch_size=4,  # Small batch size for demo
        train_seq_len=2048,
        seed=42
    )
    
    # Load tokenizer for detokenization
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    print("=== Export Handler Examples ===\n")
    
    # Example 1: Export specific batches to JSONL
    print("1. Exporting batches 0-2 to JSONL...")
    manager.export.export_batches(
        batch_ids=[0, 1, 2],
        batch_size=4,
        output_path="exports/batches_example.jsonl",
        format_type="jsonl",
        return_detokenized=True,
        tokenizer=tokenizer,
        include_doc_details=True,
        flatten_batches=True
    )
    print("   ✓ Exported to exports/batches_example.jsonl\n")
    
    # Example 2: Export specific sequences to CSV
    print("2. Exporting sequences 0-9 to CSV...")
    manager.export.export_sequences(
        sequence_indices=list(range(10)),
        output_path="exports/sequences_example.csv",
        format_type="csv",
        return_detokenized=True,
        tokenizer=tokenizer,
        include_doc_details=False
    )
    print("   ✓ Exported to exports/sequences_example.csv\n")
    
    # Example 3: Export small sample of entire dataset to JSONL (first 50 samples)
    print("3. Exporting first 50 samples of dataset to JSONL...")
    # For demo purposes, we'll simulate exporting a small portion
    manager.export.export_sequences(
        sequence_indices=list(range(50)),
        output_path="exports/dataset_sample.jsonl",
        format_type="jsonl",
        return_detokenized=True,
        tokenizer=tokenizer,
        include_doc_details=True
    )
    print("   ✓ Exported to exports/dataset_sample.jsonl\n")
    
    # Example 4: Export raw tokens (not detokenized) to show format difference
    print("4. Exporting raw tokens (not detokenized) to JSONL...")
    manager.export.export_sequences(
        sequence_indices=list(range(5)),
        output_path="exports/raw_tokens_example.jsonl",
        format_type="jsonl",
        return_detokenized=False,
        tokenizer=None,
        include_doc_details=True
    )
    print("   ✓ Exported to exports/raw_tokens_example.jsonl\n")
    
    # Example 5: Export range of sequences (new feature)
    print("5. Exporting sequence range 100-149 to JSONL...")
    manager.export.export_sequence_range(
        start_idx=100,
        end_idx=150,  # Exports sequences 100-149
        output_path="exports/sequence_range_example.jsonl",
        format_type="jsonl",
        return_detokenized=True,
        tokenizer=tokenizer,
        include_doc_details=True
    )
    print("   ✓ Exported to exports/sequence_range_example.jsonl\n")
    
    # Example 6: Export range of batches (new feature)
    print("6. Exporting batch range 5-9 to CSV...")
    manager.export.export_batch_range(
        start_batch=5,
        end_batch=10,  # Exports batches 5-9
        batch_size=4,
        output_path="exports/batch_range_example.csv",
        format_type="csv",
        return_detokenized=True,
        tokenizer=tokenizer,
        include_doc_details=False,
        flatten_batches=True
    )
    print("   ✓ Exported to exports/batch_range_example.csv\n")
    
    # Example 7: Export dataset range with chunking (new feature)
    print("7. Exporting dataset range 1000-1999 with chunking...")
    manager.export.export_dataset_range(
        start_idx=1000,
        end_idx=2000,  # Exports sequences 1000-1999
        output_path="exports/dataset_range_example.jsonl",
        format_type="jsonl",
        return_detokenized=True,
        tokenizer=tokenizer,
        include_doc_details=True,
        chunk_size=250  # Process 250 samples at a time
    )
    print("   ✓ Exported to exports/dataset_range_example.jsonl\n")
    
    print("=== Export Examples Complete ===")
    print("\nCheck the 'exports/' directory for the generated files.")
    print("\n=== Range-Based Export Methods ===")
    print("New range-based methods provide convenient ways to export contiguous ranges:")
    print("• export_sequence_range(start_idx, end_idx, ...)")
    print("• export_batch_range(start_batch, end_batch, batch_size, ...)")
    print("• export_dataset_range(start_idx, end_idx, ..., chunk_size)")
    print("\nNote: To export the entire dataset, use:")
    print("manager.export.export_entire_dataset(")
    print("    output_path='exports/full_dataset.jsonl',")
    print("    format_type='jsonl',")
    print("    return_detokenized=True,")
    print("    tokenizer=tokenizer,")
    print("    chunk_size=1000")
    print(")")

if __name__ == "__main__":
    main()
