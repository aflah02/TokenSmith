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
- **📤 Export Utilities**: Export data in multiple formats (JSONL, CSV)
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
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/aflah02/tokensmith.git
cd tokensmith
pip install -e .
```

### Basic Usage

```python
from tokensmith import DatasetManager
from transformers import AutoTokenizer

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create dataset manager
manager = DatasetManager()

# Setup for editing, inspection, sampling, and export
manager.setup_edit_inspect_sample_export(
    dataset_prefix="path/to/your/dataset",
    batch_info_save_prefix="path/to/batch/info",
    train_iters=1000,
    train_batch_size=32,
    train_seq_len=1024,
    seed=42
)

# Setup search functionality
manager.setup_search(
    bin_file_path="path/to/dataset.bin",
    search_index_save_path="path/to/search/index",
    vocab=2**16
)

# Now you can use all handlers
sample = manager.inspect.inspect_sample_by_id(0)
results = manager.search.search([token_id_1, token_id_2])
```

## 📚 API Documentation

This documentation contains comprehensive API reference for all TokenSmith components:

- **[DatasetManager](reference/manager.md)**: Central manager coordinating all handlers
- **[Search Handler](reference/search.md)**: Token sequence search and indexing
- **[Inspect Handler](reference/inspect.md)**: Dataset examination and visualization
- **[Sample Handler](reference/sample.md)**: Flexible data sampling strategies
- **[Edit Handler](reference/edit.md)**: Dataset modification and injection
- **[Export Handler](reference/export.md)**: Multi-format data export
- **[Ingest Handler](reference/ingest.md)**: Multi-format data ingestion
- **[Utilities](reference/utils.md)**: Utility functions and classes
- **[UI Components](reference/ui.md)**: Interactive Streamlit interface

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for more information.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
