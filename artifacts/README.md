# Artifacts Directory

This directory contains various data files, processing scripts, and outputs generated during the TokenSmith data processing pipeline. The artifacts serve as examples and test data for demonstrating the capabilities of the TokenSmith toolkit.

## 📁 Directory Structure

### 📊 Raw Data Files

- **`TinyStories-valid.txt`** - Validation dataset from the TinyStories corpus containing simple children's stories. This serves as the primary text corpus for testing and demonstration purposes.
- **`data.jsonl`** - Processed dataset in JSON Lines format, containing individual stories as structured text records. Each line contains a JSON object with a "text" field.

### 🔧 Processing Scripts

- **`tokenize.sh`** - Bash script for tokenizing the dataset using GPT-NeoX preprocessing tools

### 🎯 Tokenizer Configuration

- **`tokenizer.json`** - HuggingFace tokenizer configuration file (GPT-NeoX 20B tokenizer)
  - Vocabulary size: 50,277 tokens (padded to 50,304)
  - Used for consistent tokenization across the pipeline

### Tokenized Dataset Files
- **`data_tokenized_text_document.bin`** - Binary file containing tokenized text data
- **`data_tokenized_text_document.idx`** - Index file for the tokenized binary data

These above files correspond to the outputs produced by `tokenize.sh` when operated over `data.jsonl`

### 📝 Log Files

- **`tokenize.log`** - Simple output log from the tokenize.sh script execution

### 📓 Jupyter Notebooks

- **`tokensmith_showcase.ipynb`** - Basic showcase of TokenSmith capabilities. Used for the Demo Video
- **`prepare_dataset.ipynb`** - Data preparation and preprocessing examples


## 🔄 Data Processing Pipeline

The artifacts follow this typical processing flow (Files in Step 3-5 are generated and not kept in the repo):

1. **Raw Text** (`TinyStories-valid.txt`) → **Structured Data** (`data.jsonl`)
2. **Tokenization** (`tokenize.sh`) → **Binary Format** (`data_tokenized_*`)
3. **Ingestion** → **Processed Data** (`data_ingested_*`)
4. **Indexing** → **Search Indices** (`search_index_*`)
5. **Batch Preparation** → **Training Metadata** (`batch_info_*`)

## 🚀 Usage Examples

### Running Tokenization
```bash
cd artifacts
chmod +x tokenize.sh
./tokenize.sh
```