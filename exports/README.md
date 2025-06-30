# Export Handler Documentation

The Export Handler provides functionality to export dataset samples in different formats and granularities.

## Features

### Export Types
1. **Export Batches**: Export specific training batches
2. **Export Sequences**: Export individual sequences by their indices
3. **Export Entire Dataset**: Export the complete dataset with memory-efficient chunking
4. **Export Sequence Range**: Export a contiguous range of sequences (NEW)
5. **Export Batch Range**: Export a contiguous range of batches (NEW)
6. **Export Dataset Range**: Export a contiguous range of the dataset with chunking (NEW)

### Output Formats
- **JSONL**: JSON Lines format (one JSON object per line)
- **CSV**: Comma-separated values format

### Export Options
- **Detokenized Text**: Export human-readable text (requires tokenizer)
- **Raw Tokens**: Export token arrays as lists of integers
- **Document Details**: Include metadata about source documents
- **Batch Flattening**: Option to flatten batches into single sequence list

## Usage Examples

### Basic Setup
```python
from tokensmith import DatasetManager
from transformers import AutoTokenizer

manager = DatasetManager()
manager.setup_edit_inspect_sample_export(
    dataset_prefix="path/to/dataset",
    batch_info_save_prefix="path/to/batch_info",
    train_iters=1600,
    train_batch_size=32,
    train_seq_len=2048,
    seed=42
)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

### Export Batches
```python
# Export specific batches to JSONL
manager.export.export_batches(
    batch_ids=[0, 1, 2],
    batch_size=32,
    output_path="exports/batches.jsonl",
    format_type="jsonl",
    return_detokenized=True,
    tokenizer=tokenizer,
    include_doc_details=True,
    flatten_batches=True
)
```

### Export Sequences
```python
# Export specific sequences to CSV
manager.export.export_sequences(
    sequence_indices=[100, 200, 300],
    output_path="exports/sequences.csv",
    format_type="csv",
    return_detokenized=True,
    tokenizer=tokenizer,
    include_doc_details=False
)
```

### Export Entire Dataset
```python
# Export entire dataset with chunking for memory efficiency
manager.export.export_entire_dataset(
    output_path="exports/full_dataset.jsonl",
    format_type="jsonl",
    return_detokenized=True,
    tokenizer=tokenizer,
    include_doc_details=True,
    chunk_size=1000  # Process 1000 samples at a time
)
```

### Export Sequence Range (NEW)
```python
# Export a contiguous range of sequences
manager.export.export_sequence_range(
    start_idx=100,
    end_idx=200,  # Exports sequences 100-199
    output_path="exports/sequence_range.jsonl",
    format_type="jsonl",
    return_detokenized=True,
    tokenizer=tokenizer,
    include_doc_details=True
)
```

### Export Batch Range (NEW)
```python
# Export a contiguous range of batches
manager.export.export_batch_range(
    start_batch=5,
    end_batch=10,  # Exports batches 5-9
    batch_size=32,
    output_path="exports/batch_range.csv",
    format_type="csv",
    return_detokenized=True,
    tokenizer=tokenizer,
    flatten_batches=True
)
```

### Export Dataset Range (NEW)
```python
# Export a range of the dataset with memory-efficient chunking
manager.export.export_dataset_range(
    start_idx=1000,
    end_idx=5000,  # Exports sequences 1000-4999
    output_path="exports/dataset_range.jsonl",
    format_type="jsonl",
    return_detokenized=True,
    tokenizer=tokenizer,
    include_doc_details=True,
    chunk_size=500  # Process 500 samples at a time
)
```

## Output Format Examples

### JSONL Format (with document details)
```json
{"index": 0, "content": "This is the detokenized text content.", "doc_details": {"doc_index_f": 0, "doc_index_l": 1, "offset_f": 0, "offset_l": 2048}}
{"index": 1, "content": "Another sample of text content.", "doc_details": {"doc_index_f": 1, "doc_index_l": 2, "offset_f": 1024, "offset_l": 3072}}
```

### CSV Format
```csv
index,content
0,"This is the detokenized text content."
1,"Another sample of text content."
```

### Raw Tokens Format
```json
{"index": 0, "content": [15496, 318, 262, 1334, 1134, 7496, 2420, 1239, 13]}
```

## Memory Considerations

- Use `chunk_size` parameter for large dataset exports to control memory usage
- Consider using `return_detokenized=False` for faster exports when human-readable text is not needed
- The `flatten_batches` option can help organize data for downstream processing
- Range-based exports are efficient for exporting contiguous sections of data
- Dataset range exports use chunking automatically for memory efficiency

## Error Handling

The Export Handler includes comprehensive validation:
- Validates format types (only "jsonl" and "csv" supported)
- Ensures tokenizer is provided when detokenization is requested
- Creates output directories automatically if they don't exist
- Handles file encoding properly for international text
- Validates range parameters (start < end, non-negative indices)
- Checks dataset bounds for range exports
