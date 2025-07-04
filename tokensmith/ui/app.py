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
    parser.add_argument("--reuse-index", action="store_true", help="Reuse existing search index if available")
    
    # Dataset arguments
    parser.add_argument("--dataset-prefix", type=str, help="Prefix for the dataset files")
    parser.add_argument("--batch-info-prefix", type=str, help="Prefix for the batch information files")
    parser.add_argument("--train-iters", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--train-seq-len", type=int, default=1024, help="Training sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--splits", type=str, default="969,30,1", help="Train/val/test splits")
    parser.add_argument("--packing-impl", type=str, default="packed", choices=["packed", "pack_until_overflow", "unpacked"], help="Packing implementation")
    parser.add_argument("--allow-chopped", action="store_true", help="Allow chopped samples")
    parser.add_argument("--extra-tokens", type=int, default=1, help="Extra tokens to add to sequence")
    
    # Tokenizer arguments
    parser.add_argument("--tokenizer-path", type=str, help="Path to tokenizer for detokenization")
    
    # Mode argument
    parser.add_argument("--mode", type=str, choices=["search", "inspect", "both"], default="both", 
                       help="UI mode: 'search' for search only, 'inspect' for inspect and view documents, 'both' for all features")
    
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
                mode = "both"
            st.session_state.args = DefaultArgs()
            
    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = None
        # Try to load tokenizer if path provided
        if hasattr(st.session_state.args, 'tokenizer_path') and st.session_state.args.tokenizer_path:
            try:
                from transformers import AutoTokenizer
                print(f"Loading tokenizer from {st.session_state.args.tokenizer_path}")
                st.session_state.tokenizer = AutoTokenizer.from_pretrained(st.session_state.args.tokenizer_path)
            except Exception as e:
                st.error(f"Failed to load tokenizer: {e}")
    
    # Initialize handlers based on mode
    mode = getattr(st.session_state.args, 'mode', 'both')
    
    if mode in ["search", "both"]:
        # Initialize search if required arguments are provided
        if (hasattr(st.session_state.args, 'bin_file_path') and st.session_state.args.bin_file_path and
            hasattr(st.session_state.args, 'search_index_path') and st.session_state.args.search_index_path and
            hasattr(st.session_state.args, 'vocab') and st.session_state.args.vocab):
            try:
                if 'search_setup_done' not in st.session_state:
                    st.session_state.search_setup_done = False
                if not st.session_state.search_setup_done:
                    st.session_state.search_setup_done = True
                    # Initialize search handler
                    print("Reuse Index:", getattr(st.session_state.args, 'reuse_index', False))
                    st.session_state.dataset_manager.setup_search(
                        bin_file_path=st.session_state.args.bin_file_path,
                        search_index_save_path=st.session_state.args.search_index_path,
                        vocab=st.session_state.args.vocab,
                        verbose=getattr(st.session_state.args, 'search_verbose', False),
                        reuse=getattr(st.session_state.args, 'reuse_index', False)
                    )
                else:
                    # Already set up, no need to reinitialize
                    pass
            except Exception as e:
                st.error(f"Failed to initialize search: {e}")
    
    if mode in ["inspect", "both"]:
        # Initialize inspect if required arguments are provided
        if (hasattr(st.session_state.args, 'dataset_prefix') and st.session_state.args.dataset_prefix and
            hasattr(st.session_state.args, 'batch_info_prefix') and st.session_state.args.batch_info_prefix):
            try:
                if 'inspect_setup_done' not in st.session_state:
                    st.session_state.inspect_setup_done = False
                if not st.session_state.inspect_setup_done:
                    st.session_state.dataset_manager.setup_edit_inspect_sample_export(
                        dataset_prefix=st.session_state.args.dataset_prefix,
                        batch_info_save_prefix=st.session_state.args.batch_info_prefix,
                        train_iters=getattr(st.session_state.args, 'train_iters', 1000),
                        train_batch_size=getattr(st.session_state.args, 'train_batch_size', 16),
                        train_seq_len=getattr(st.session_state.args, 'train_seq_len', 1024),
                        seed=getattr(st.session_state.args, 'seed', 42),
                        splits_string=getattr(st.session_state.args, 'splits', '969,30,1'),
                        packing_impl=getattr(st.session_state.args, 'packing_impl', 'packed'),
                        allow_chopped=getattr(st.session_state.args, 'allow_chopped', True),
                        add_extra_token_to_seq=getattr(st.session_state.args, 'extra_tokens', 1)
                    )
                    st.session_state.inspect_setup_done = True
                else:
                    pass  # Already set up, no need to reinitialize
            except Exception as e:
                st.error(f"Failed to initialize inspect: {e}")

# Initialize session state
init_session_state()

# Create pages based on available functionality
pages = []
mode = getattr(st.session_state.args, 'mode', 'both')

if mode in ["inspect", "both"] and st.session_state.dataset_manager.inspect is not None:
    inspect_page = st.Page("pages/inspect.py", title="Inspect Dataset", icon=":material/eye_tracking:")
    pages.append(inspect_page)
    
    # Add view documents page when inspect is available
    view_documents_page = st.Page("pages/view_documents.py", title="View Documents", icon=":material/description:")
    pages.append(view_documents_page)

if mode in ["search", "both"] and st.session_state.dataset_manager.search is not None:
    search_page = st.Page("pages/search.py", title="Search Dataset", icon=":material/find_in_page:")
    pages.append(search_page)

# If no pages are available, show an error
if not pages:
    st.error("❌ No functionality initialized. Please provide the required CLI arguments for your desired mode.")
    st.write("Available modes:")
    st.write("- **search**: Requires `--bin-file-path`, `--search-index-path`, and `--vocab`")
    st.write("- **inspect**: Requires `--dataset-prefix` and `--batch-info-prefix` (includes Inspect Dataset and View Documents)")
    st.write("- **both**: Requires arguments for both modes")
    st.stop()

pg = st.navigation(pages)
st.set_page_config(page_title="TokenSmith UI", page_icon=":material/key:", layout="wide")

pg.run()
