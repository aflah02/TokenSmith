#!/bin/bash

# TokenSmith UI Runner Script
# This script helps you run the TokenSmith Streamlit UI with the necessary arguments

# Default values (update these based on your setup)
BIN_FILE_PATH="/NS/llm-pretraining/work/afkhan/tokensmith/artifacts/data_tokenized_text_document.bin"
SEARCH_INDEX_PATH="/NS/llm-pretraining/work/afkhan/tokensmith/artifacts/search_index_text_document.idx"
VOCAB=65536  # 2^16
DATASET_PREFIX="/NS/llm-pretraining/work/afkhan/tokensmith/artifacts/data_tokenized_text_document"
BATCH_INFO_PREFIX="/NS/llm-pretraining/work/afkhan/tokensmith/artifacts/batch_info_train_indexmap_1600ns_2048sl_42s_packedpi_ac"
TOKENIZER_PATH="/NS/llm-pretraining/work/afkhan/tokensmith/artifacts/tokenizer.json"

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run TokenSmith Streamlit UI with optional configuration"
    echo ""
    echo "Options:"
    echo "  --bin-file-path PATH        Path to binary dataset file"
    echo "  --search-index-path PATH    Path to search index file"
    echo "  --vocab SIZE               Vocabulary size (65536 or 4294967296)"
    echo "  --dataset-prefix PATH      Dataset prefix"
    echo "  --batch-info-prefix PATH   Batch info prefix"
    echo "  --tokenizer-path PATH      Path to tokenizer"
    echo "  --port PORT                Streamlit port (default: 8501)"
    echo "  --help                     Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --bin-file-path /path/to/data.bin --vocab 65536"
}

# Parse command line arguments
PORT=8501
while [[ $# -gt 0 ]]; do
    case $1 in
        --bin-file-path)
            BIN_FILE_PATH="$2"
            shift 2
            ;;
        --search-index-path)
            SEARCH_INDEX_PATH="$2"
            shift 2
            ;;
        --vocab)
            VOCAB="$2"
            shift 2
            ;;
        --dataset-prefix)
            DATASET_PREFIX="$2"
            shift 2
            ;;
        --batch-info-prefix)
            BATCH_INFO_PREFIX="$2"
            shift 2
            ;;
        --tokenizer-path)
            TOKENIZER_PATH="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if required files exist
echo "Checking configuration..."
echo "Binary file: $BIN_FILE_PATH"
echo "Search index: $SEARCH_INDEX_PATH"
echo "Dataset prefix: $DATASET_PREFIX"
echo "Batch info prefix: $BATCH_INFO_PREFIX"
echo "Tokenizer: $TOKENIZER_PATH"
echo "Vocabulary size: $VOCAB"
echo ""

# Build the streamlit command
STREAMLIT_CMD="streamlit run app.py --server.port $PORT --"

# Add arguments
STREAMLIT_CMD="$STREAMLIT_CMD --bin-file-path '$BIN_FILE_PATH'"
STREAMLIT_CMD="$STREAMLIT_CMD --search-index-path '$SEARCH_INDEX_PATH'"
STREAMLIT_CMD="$STREAMLIT_CMD --vocab $VOCAB"
STREAMLIT_CMD="$STREAMLIT_CMD --dataset-prefix '$DATASET_PREFIX'"
STREAMLIT_CMD="$STREAMLIT_CMD --batch-info-prefix '$BATCH_INFO_PREFIX'"
STREAMLIT_CMD="$STREAMLIT_CMD --tokenizer-path '$TOKENIZER_PATH'"
STREAMLIT_CMD="$STREAMLIT_CMD --reuse-index"
STREAMLIT_CMD="$STREAMLIT_CMD --allow-chopped"

echo "Starting TokenSmith UI..."
echo "Command: $STREAMLIT_CMD"
echo ""
echo "The UI will be available at: http://localhost:$PORT"
echo ""

# Change to the UI directory and run
cd "$(dirname "$0")"
eval $STREAMLIT_CMD
