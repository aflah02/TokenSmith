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

TokenSmith can be installed with different dependency sets depending on your use case:

#### Option 1: Core Dependencies Only

For basic installation (most functionality still requires GPT-NeoX environment):

```bash
git clone https://github.com/aflah02/tokensmith.git
cd tokensmith
pip install -e .
```

This installs core dependencies:
- `numpy` - Array operations
- `pandas` - Data processing  
- `tqdm` - Progress bars

**Note:** This installation alone only allows imports. Dataset operations, UI, and most functionality require GPT-NeoX environment.

#### Option 2: With Search Functionality (Works Standalone)

For search and indexing operations using tokengrams - **this is the only option that works without GPT-NeoX**:

```bash
git clone https://github.com/aflah02/tokensmith.git
cd tokensmith
pip install -e ".[search]"
```

Or with Poetry:
```bash
poetry install --with search
```

#### Option 3: With UI Support (Requires GPT-NeoX)

For the interactive Streamlit web interface:

```bash
git clone https://github.com/aflah02/tokensmith.git
cd tokensmith
pip install -e ".[ui]"
```

Or with Poetry:
```bash
poetry install --with ui
```

#### Option 4: With Documentation Tools

For building documentation:

```bash
pip install -e ".[docs]"
```

Or with Poetry:
```bash
poetry install --with docs
```

#### Option 5: Complete Installation (Requires GPT-NeoX)

For all optional dependencies (search, UI, and docs):

```bash
pip install -e ".[all]"
```

Or with Poetry:
```bash
poetry install --with all
```

You can also combine multiple options:
```bash
pip install -e ".[search,ui]"  # Search + UI
```

#### GPT-NeoX/Megatron Integration

**Note:** For functionality that requires GPT-NeoX/Megatron (such as `WriteableMMapIndexedDataset` and some advanced dataset operations), you must separately install GPT-NeoX following the steps provided [here](https://github.com/EleutherAI/gpt-neox?tab=readme-ov-file#environment-and-dependencies).

TokenSmith is designed to work with or without GPT-NeoX:
- **Without GPT-NeoX**: Only search functionality works standalone
- **With GPT-NeoX**: Full functionality including UI, dataset operations, editing, sampling, and advanced operations

**Note:** `torch` and `transformers` are provided by the GPT-NeoX environment and are not included as TokenSmith dependencies to avoid version conflicts.

#### Python Version Requirements

- **Python 3.8+** is required
- Compatible with modern Python versions and dependency ecosystems

#### Which Installation Option to Choose?

- **Search only**: Use Option 2 if you only need token sequence search and indexing (works standalone)
- **Full functionality**: Use Options 3-5 if you need UI or dataset operations (requires GPT-NeoX environment)
  - **Web interface**: Use Option 3 for interactive Streamlit UI
  - **Documentation**: Use Option 4 for contributing to docs
  - **Complete features**: Use Option 5 for all functionality
- **Development**: Use Option 5 for developing TokenSmith

**Important**: Only search functionality works without GPT-NeoX. All other features require the GPT-NeoX environment.

### Basic Usage

#### Search Functionality (Works standalone - no GPT-NeoX required)

```python
from tokensmith import DatasetManager

# Initialize the manager
manager = DatasetManager()

# Setup search functionality - requires tokengrams but no GPT-NeoX
try:
    manager.setup_search(
        bin_file_path="path/to/dataset.bin",
        search_index_save_path="path/to/search_index",
        vocab=2**16,  # or 2**32 for larger vocabularies
        reuse=True
    )
    
    # Search operations
    query = [101, 2023, 102]  # Token IDs
    count = manager.search.count(query)
    positions = manager.search.positions(query)
    print("✅ Search functionality available")
    
except ImportError as e:
    print("ℹ️ Search functionality requires tokengrams: pip install 'tokensmith[search]'")
```

#### Dataset Operations (Requires GPT-NeoX environment)

```python
from tokensmith import DatasetManager

# Initialize the manager
manager = DatasetManager()

# Note: All operations below require GPT-NeoX environment to be installed
try:
    # Setup dataset for inspection, sampling, editing, and export
    manager.setup_edit_inspect_sample_export(
        dataset_prefix="path/to/your/dataset",
        batch_info_save_prefix="path/to/batch_info",
        train_iters=1000,
        train_batch_size=32,
        train_seq_len=1024,
        seed=42
    )
    
    # Load a tokenizer (requires transformers from GPT-NeoX environment)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    print("✅ Full functionality available")
except ImportError as e:
    print("ℹ️ Dataset operations require GPT-NeoX installation")
    print("   See: https://github.com/EleutherAI/gpt-neox")
```

#### Web UI Usage (Requires GPT-NeoX environment)

```bash
# Note: UI requires GPT-NeoX environment
# Navigate to UI directory and run
cd tokensmith/ui
./run_ui.sh

# Or modify run_ui.sh for your specific setup
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
- **View Documents Page**: View individual documents in training or corpus order

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
        └── view_documents.py
```

## 📖 Documentation

### API Reference

Complete API documentation with automatically generated docstrings is available at:
**[https://aflah02.github.io/TokenSmith](https://aflah02.github.io/TokenSmith)**

### Tutorials

Comprehensive tutorials and examples are available in the `tutorials/` directory:

- **[Basic Setup Tutorial](docs/tutorials/01_basic_setup.ipynb)** 
- **[Dataset Inspection Tutorial](docs/tutorials/02_inspect_samples.ipynb)** 
- **[Dataset Sampling Tutorial](docs/tutorials/03_sampling_methods.ipynb)**
- **[Dataset Editing Tutorial](docs/tutorials/04_dataset_editing_methods.ipynb)**
- **[Dataset Searching Tutorial](docs/tutorials/05_search_functionality.ipynb)**


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
- 📖 **Documentation**: [https://aflah02.github.io/TokenSmith](https://aflah02.github.io/TokenSmith)

## ℹ️ Citation

If you find this library useful or build upon it, please remember to cite our work -

```
@misc{khan2025tokensmithstreamliningdataediting,
      title={TokenSmith: Streamlining Data Editing, Search, and Inspection for Large-Scale Language Model Training and Interpretability}, 
      author={Mohammad Aflah Khan and Ameya Godbole and Johnny Tian-Zheng Wei and Ryan Wang and James Flemings and Krishna Gummadi and Willie Neiswanger and Robin Jia},
      year={2025},
      eprint={2507.19419},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.19419}, 
}
```
