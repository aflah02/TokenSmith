import streamlit as st
import argparse
import sys
import os

# Add parent directory to path to import tokensmith
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokensmith.manager import DatasetManager

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="TokenSmith Streamlit UI")
    
    # Search arguments
    parser.add_argument("--bin-file-path", type=str, help="Path to the binary file containing the dataset")
    parser.add_argument("--search-index-path", type=str, help="Path to save/load the search index")
    parser.add_argument("--vocab", type=int, choices=[2**16, 2**32], help="Vocabulary size (65536 or 4294967296)")
    parser.add_argument("--search-verbose", action="store_true", help="Enable verbose output for search index building")
    parser.add_argument("--reuse-index", action="store_true", default=True, help="Reuse existing search index if available")
    
    # Dataset arguments
    parser.add_argument("--dataset-prefix", type=str, help="Prefix for the dataset files")
    parser.add_argument("--batch-info-prefix", type=str, help="Prefix for the batch information files")
    parser.add_argument("--train-iters", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--train-seq-len", type=int, default=1024, help="Training sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--splits", type=str, default="969,30,1", help="Train/val/test splits")
    parser.add_argument("--packing-impl", type=str, default="packed", choices=["packed", "pack_until_overflow", "unpacked"], help="Packing implementation")
    parser.add_argument("--allow-chopped", action="store_true", default=True, help="Allow chopped samples")
    parser.add_argument("--extra-tokens", type=int, default=1, help="Extra tokens to add to sequence")
    
    # Tokenizer arguments
    parser.add_argument("--tokenizer-path", type=str, help="Path to tokenizer for detokenization")
    
    return parser.parse_args()

# Initialize session state with DatasetManager
def init_session_state():
    if "dataset_manager" not in st.session_state:
        st.session_state.dataset_manager = DatasetManager()
        
    if "args" not in st.session_state:
        # Try to get command line args, fallback to defaults if running in Streamlit Cloud
        try:
            st.session_state.args = parse_args()
        except SystemExit:
            # Fallback for Streamlit Cloud or when no args provided
            class DefaultArgs:
                pass
            st.session_state.args = DefaultArgs()
            
    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = None
        # Try to load tokenizer if path provided
        if hasattr(st.session_state.args, 'tokenizer_path') and st.session_state.args.tokenizer_path:
            try:
                from transformers import AutoTokenizer
                st.session_state.tokenizer = AutoTokenizer.from_pretrained(st.session_state.args.tokenizer_path)
            except Exception as e:
                st.error(f"Failed to load tokenizer: {e}")

# Initialize session state
init_session_state()

search_page = st.Page("pages/search.py", title="Search Dataset", icon=":material/find_in_page:")
inspect_page = st.Page("pages/inspect.py", title="Inspect Dataset", icon=":material/eye_tracking:")

pg = st.navigation([search_page, inspect_page])
st.set_page_config(page_title="TokenSmith UI", page_icon=":material/key:", layout="wide")

pg.run()
