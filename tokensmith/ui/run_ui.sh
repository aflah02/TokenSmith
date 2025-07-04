#!/bin/bash

# TokenSmith UI Runner Script
# This script helps you run the TokenSmith Streamlit UI with the necessary arguments

# Default values (update these based on your setup)
BIN_FILE_PATH="../../artifacts/data_tokenized_text_document.bin"
SEARCH_INDEX_PATH="../../artifacts/search_index_text_document_2.idx"
VOCAB=65536  # 2^16
DATASET_PREFIX="../../artifacts/data_tokenized_text_document"
BATCH_INFO_PREFIX="../../artifacts/batch_info"
TOKENIZER_PATH="EleutherAI/gpt-neox-20b"
TRAIN_ITERS=1000
TRAIN_BATCH_SIZE=16
TRAIN_SEQ_LEN=1024
SEED=42
SPLITS="969,30,1"
PACKING_IMPL="packed"
EXTRA_TOKENS=1
SEARCH_VERBOSE=false
REUSE_INDEX=false
ALLOW_CHOPPED=false
MODE="both"
# MODE="inspect"

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
    echo "  --train-iters NUM          Number of training iterations (default: 1000)"
    echo "  --train-batch-size NUM     Training batch size (default: 16)"
    echo "  --train-seq-len NUM        Training sequence length (default: 1024)"
    echo "  --seed NUM                 Random seed (default: 42)"
    echo "  --splits STRING            Train/val/test splits (default: '969,30,1')"
    echo "  --packing-impl TYPE        Packing implementation: packed, pack_until_overflow, unpacked (default: packed)"
    echo "  --extra-tokens NUM         Extra tokens to add to sequence (default: 1)"
    echo "  --search-verbose           Enable verbose output for search index building"
    echo "  --reuse-index              Reuse existing search index if available"
    echo "  --allow-chopped            Allow chopped samples"
    echo "  --mode MODE                UI mode: search, inspect, or both (default: both)"
    echo "  --port PORT                Streamlit port (default: 8501)"
    echo "  --help                     Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --bin-file-path /path/to/data.bin --vocab 65536 --mode search"
    echo "  $0 --dataset-prefix /path/to/dataset --batch-info-prefix /path/to/batch --mode inspect"
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
        --train-iters)
            TRAIN_ITERS="$2"
            shift 2
            ;;
        --train-batch-size)
            TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --train-seq-len)
            TRAIN_SEQ_LEN="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --splits)
            SPLITS="$2"
            shift 2
            ;;
        --packing-impl)
            PACKING_IMPL="$2"
            shift 2
            ;;
        --extra-tokens)
            EXTRA_TOKENS="$2"
            shift 2
            ;;
        --search-verbose)
            SEARCH_VERBOSE=true
            shift
            ;;
        --reuse-index)
            REUSE_INDEX=true
            shift
            ;;
        --allow-chopped)
            ALLOW_CHOPPED=true
            shift
            ;;
        --mode)
            MODE="$2"
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
echo "Training iterations: $TRAIN_ITERS"
echo "Training batch size: $TRAIN_BATCH_SIZE"
echo "Training sequence length: $TRAIN_SEQ_LEN"
echo "Seed: $SEED"
echo "Splits: $SPLITS"
echo "Packing implementation: $PACKING_IMPL"
echo "Extra tokens: $EXTRA_TOKENS"
echo "Search verbose: $SEARCH_VERBOSE"
echo "Reuse index: $REUSE_INDEX"
echo "Allow chopped: $ALLOW_CHOPPED"
echo "Mode: $MODE"
echo ""

# Validate required arguments based on mode
case $MODE in
    "search")
        if [ -z "$BIN_FILE_PATH" ] || [ -z "$SEARCH_INDEX_PATH" ] || [ -z "$VOCAB" ]; then
            echo "Error: Search mode requires --bin-file-path, --search-index-path, and --vocab"
            exit 1
        fi
        ;;
    "inspect")
        if [ -z "$DATASET_PREFIX" ] || [ -z "$BATCH_INFO_PREFIX" ]; then
            echo "Error: Inspect mode requires --dataset-prefix and --batch-info-prefix"
            exit 1
        fi
        ;;
    "both")
        if [ -z "$BIN_FILE_PATH" ] || [ -z "$SEARCH_INDEX_PATH" ] || [ -z "$VOCAB" ] || [ -z "$DATASET_PREFIX" ] || [ -z "$BATCH_INFO_PREFIX" ]; then
            echo "Error: Both mode requires all search and inspect arguments:"
            echo "  Search: --bin-file-path, --search-index-path, --vocab"
            echo "  Inspect: --dataset-prefix, --batch-info-prefix"
            exit 1
        fi
        ;;
    *)
        echo "Error: Invalid mode '$MODE'. Must be 'search', 'inspect', or 'both'"
        exit 1
        ;;
esac
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
STREAMLIT_CMD="$STREAMLIT_CMD --train-iters $TRAIN_ITERS"
STREAMLIT_CMD="$STREAMLIT_CMD --train-batch-size $TRAIN_BATCH_SIZE"
STREAMLIT_CMD="$STREAMLIT_CMD --train-seq-len $TRAIN_SEQ_LEN"
STREAMLIT_CMD="$STREAMLIT_CMD --seed $SEED"
STREAMLIT_CMD="$STREAMLIT_CMD --splits '$SPLITS'"
STREAMLIT_CMD="$STREAMLIT_CMD --packing-impl $PACKING_IMPL"
STREAMLIT_CMD="$STREAMLIT_CMD --extra-tokens $EXTRA_TOKENS"
STREAMLIT_CMD="$STREAMLIT_CMD --mode $MODE"

# Add boolean flags if enabled
if [ "$SEARCH_VERBOSE" = true ]; then
    STREAMLIT_CMD="$STREAMLIT_CMD --search-verbose"
fi

if [ "$REUSE_INDEX" = true ]; then
    STREAMLIT_CMD="$STREAMLIT_CMD --reuse-index"
fi

if [ "$ALLOW_CHOPPED" = true ]; then
    STREAMLIT_CMD="$STREAMLIT_CMD --allow-chopped"
fi

echo "Starting TokenSmith UI..."
echo "Command: $STREAMLIT_CMD"
echo ""
echo "The UI will be available at: http://localhost:$PORT"
echo ""

# Set PYTHONPATH to include gpt-neox
export PYTHONPATH="/NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox:$PYTHONPATH"

# Change to the UI directory and run
cd "$(dirname "$0")"
eval $STREAMLIT_CMD
