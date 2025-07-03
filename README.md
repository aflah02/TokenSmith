# TokenSmith 🔧

> A comprehensive toolkit for streamlining data editing, search, and inspection for large-scale language model training and interpretability.

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

TokenSmith is a powerful Python package designed to simplify dataset management for large language model training. It provides a unified interface for editing, inspecting, searching, sampling, and exporting tokenized datasets, making it easier to work with training data at scale.

## ✨ Key Features

- **🔍 Search & Index**: Fast token sequence search with n-gram indexing
- **📊 Dataset Inspection**: Examine samples, batches, and document metadata  
- **🎯 Smart Sampling**: Flexible sampling with policy-based selection
- **✏️ Dataset Editing**: Inject and modify training samples with precision
- **📤 Export Utilities**: Export data in multiple formats
- **📩 Ingest Utilities**: Ingest data from multiple formats
- **🖥️ Interactive UI**: Streamlit-based web interface for visual exploration
- **⚡ Memory Efficient**: Chunked processing for large datasets

## 🏗️ Architecture

TokenSmith is built around a central `DatasetManager` that coordinates five specialized handlers:

```
DatasetManager
├── SearchHandler    # Token sequence search and indexing
├── InspectHandler   # Dataset examination and visualization  
├── SampleHandler    # Flexible data sampling strategies
├── EditHandler      # Dataset modification and injection
└── ExportHandler    # Multi-format data export
└── IngestHandler    # Multi-format data ingestion
```

## 🚀 Quick Start

### Installation

Ensure you have a working GPT-NeoX environment using steps provided [here](https://github.com/EleutherAI/gpt-neox?tab=readme-ov-file#environment-and-dependencies)

Within the same env run the following - 

```bash
git clone https://github.com/aflah02/tokensmith.git
cd tokensmith
pip install -e .
```

### Basic Usage

```python
from tokensmith import DatasetManager
from transformers import AutoTokenizer

# Initialize the manager
manager = DatasetManager()

# Setup dataset for inspection, sampling, editing, and export
manager.setup_edit_inspect_sample_export(
    dataset_prefix="path/to/your/dataset",
    batch_info_save_prefix="path/to/batch_info",
    train_iters=1000,
    train_batch_size=32,
    train_seq_len=1024,
    seed=42
)

# Setup search functionality (optional)
manager.setup_search(
    bin_file_path="path/to/dataset.bin",
    search_index_save_path="path/to/search_index",
    vocab=2**16,  # or 2**32 for larger vocabularies
    reuse=True
)

# Load a tokenizer for detokenization
tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

## 📚 Core Functionality

### 🔍 Search Operations

```python
# Search for token sequences
query = [101, 2023, 102]  # Token IDs
count = manager.search.count(query)
positions = manager.search.positions(query)
contains = manager.search.contains(query)

# Get next token distributions
next_tokens = manager.search.count_next(query)
```

### 📊 Dataset Inspection

```python
# Inspect individual samples
sample = manager.inspect.inspect_sample_by_id(
    sample_id=42,
    return_detokenized=True,
    tokenizer=tokenizer,
    return_doc_details=True
)

# Inspect entire batches
batch = manager.inspect.inspect_sample_by_batch(
    batch_id=0,
    batch_size=32,
    return_detokenized=True,
    tokenizer=tokenizer
)
```

### 🎯 Smart Sampling

```python
# Sample by specific indices
samples = manager.sample.get_samples_by_indices(
    indices=[1, 5, 10, 42],
    return_detokenized=True,
    tokenizer=tokenizer
)

# Sample batches by ID
batches = manager.sample.get_batches_by_ids(
    batch_ids=[0, 1, 2],
    batch_size=32,
    return_detokenized=True,
    tokenizer=tokenizer
)

# Policy-based sampling
def random_policy(n_samples):
    import random
    return random.sample(range(1000), n_samples)

policy_samples = manager.sample.get_samples_by_policy(
    policy_fn=random_policy,
    n_samples=10,
    return_detokenized=True,
    tokenizer=tokenizer
)
```

### ✏️ Dataset Editing

```python
# Inject text into specific locations
manager.edit.inject_and_preview(
    text="This is injected content",
    tokenizer=tokenizer,
    injection_loc=100,
    injection_type="seq_shuffle",  # or "seq_start"
    dry_run=False
)
```

### 📤 Data Export

```python
# Export specific batches
manager.export.export_batches(
    batch_ids=[0, 1, 2],
    batch_size=32,
    output_path="exports/batches.jsonl",
    format_type="jsonl",
    return_detokenized=True,
    tokenizer=tokenizer,
    include_doc_details=True
)

# Export sequence ranges
manager.export.export_sequence_range(
    start_idx=0,
    end_idx=1000,
    output_path="exports/sequences.csv",
    format_type="csv",
    return_detokenized=True,
    tokenizer=tokenizer
)

# Export entire dataset (in chunks)
manager.export.export_entire_dataset(
    output_path="exports/full_dataset.jsonl",
    format_type="jsonl",
    return_detokenized=True,
    tokenizer=tokenizer,
    chunk_size=1000
)
```

## 🖥️ Interactive Web UI

TokenSmith includes a Streamlit-based web interface for visual dataset exploration:

```bash
# Launch the web UI using the convenience script
cd tokensmith/ui
./run_ui.sh
```

Modify `run_ui.sh` to change modes and args

The web interface provides:
- **Search Page**: Interactive token sequence search with visualization
- **Inspect Page**: Browse and examine dataset samples and batches

## 🗂️ Project Structure

```
tokensmith/
├── manager.py              # Central DatasetManager class
├── utils.py                # Utility functions and classes
├── edit/                   # Dataset editing functionality
│   └── handler.py
├── inspect/                # Dataset inspection tools
│   └── handler.py
├── search/                 # Search and indexing
│   └── handler.py
├── sample/                 # Sampling strategies
│   └── handler.py
├── export/                 # Data export utilities
│   └── handler.py
├── ingest/                 # Data ingestion utilities
│   └── handler.py
└── ui/                     # Streamlit web interface
    ├── app.py
    └── pages/
        ├── search.py
        └── inspect.py
```

## 📖 Documentation

### API Reference

Complete API documentation with automatically generated docstrings is available at:
**[https://aflah02.github.io/tokensmith](https://aflah02.github.io/tokensmith)**

### Tutorials

Comprehensive tutorials and examples are available in the `tutorials/` directory:

- **[Basic Setup](tutorials/01_basic_setup.ipynb)** - Getting started guide
- **[Dataset Inspection](tutorials/)** - Examining your data
- **[Sampling Methods](tutorials/)** - Different ways to sample data
- **[Policy-based Sampling](tutorials/)** - Advanced sampling strategies

### Building Documentation Locally

To build and serve the documentation locally:

```bash
# Install documentation dependencies
pip install -r docs-requirements.txt

# Serve locally (auto-reloads on changes)
mkdocs serve
# or use the convenience script
./serve-docs.sh
```

The documentation will be available at `http://127.0.0.1:8000`

## 🔧 Configuration

### Dataset Setup Parameters

- `dataset_prefix`: Path prefix for dataset files (.bin/.idx)
- `batch_info_save_prefix`: Path prefix for batch index files
- `train_iters`: Number of training iterations to simulate
- `train_batch_size`: Batch size for training simulation
- `train_seq_len`: Sequence length for training
- `seed`: Random seed for reproducibility
- `splits_string`: Train/val/test split ratios (default: "969,30,1")
- `packing_impl`: Sequence packing method ("packed", "pack_until_overflow", "unpacked")

### Search Setup Parameters

- `bin_file_path`: Path to binary dataset file
- `search_index_save_path`: Path for search index storage
- `vocab`: Vocabulary size (2^16 or 2^32)
- `reuse`: Whether to reuse existing index files

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache 2.0 License - see [this](https://www.apache.org/licenses/LICENSE-2.0) for further details.

## 🙏 Acknowledgments

- Built on top of the [tokengrams](https://github.com/EleutherAI/tokengrams) library for efficient n-gram indexing
- Uses Megatron-style dataset indexing for compatibility with existing training pipelines

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/aflah02/tokensmith/issues)
- 📖 **Documentation**: [https://aflah02.github.io/tokensmith](https://aflah02.github.io/tokensmith)
