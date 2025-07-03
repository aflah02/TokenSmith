import streamlit as st
import json
import numpy as np

st.title("View Documents")

# Check if inspect handler is initialized
if not hasattr(st.session_state, 'dataset_manager') or st.session_state.dataset_manager.inspect is None:
    st.error("❌ Document viewer functionality not initialized. Please restart the app with proper CLI arguments.")
    st.write("Required arguments:")
    st.code("""
streamlit run app.py -- \\
    --dataset-prefix /path/to/dataset \\
    --batch-info-prefix /path/to/batch_info \\
    --train-iters 1000 \\
    --train-batch-size 16 \\
    --train-seq-len 1024 \\
    --seed 42
    """)
    st.stop()

# Main interface
st.sidebar.title("Document Viewer Settings")

# Mode selection
document_mode = st.sidebar.radio(
    "Ordering to Follow:",
    ["Training Order", "Corpus Order"],
    help="Choose whether to retrieve documents in the order they will be presented during training or as they are stored in the corpus"
)

# Document ID input
document_id = st.sidebar.number_input(
    "Document ID:", 
    min_value=0, 
    value=0, 
    step=1,
    help="Enter the document ID to view"
)

# Detokenization options
return_detokenized = st.sidebar.checkbox(
    "Return detokenized text", 
    value=False,
    help="Convert tokens back to readable text"
)

if return_detokenized and not st.session_state.tokenizer:
    st.sidebar.warning("Tokenizer not available. Please provide tokenizer path in CLI args or disable detokenization.")
    return_detokenized = False

# Display options
show_raw_tokens = st.sidebar.checkbox(
    "Show raw tokens", 
    value=True,
    help="Display the raw token array"
)



# Action buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("View Document", type="primary"):
        try:
            with st.spinner("Fetching document..."):
                # Get the document using the appropriate method
                if document_mode == "Corpus Order":
                    document_data = st.session_state.dataset_manager.WriteableMMapIndexedDataset.get_corpus_document_by_id(document_id)
                    st.success(f"✅ Successfully retrieved document {document_id} in corpus order")
                else:  # Training Order
                    document_data = st.session_state.dataset_manager.WriteableMMapIndexedDataset.get_train_document_by_id(document_id)
                    st.success(f"✅ Successfully retrieved document {document_id} in training order")
                
                # Store results in session state for persistence
                st.session_state.current_document = {
                    'data': document_data,
                    'id': document_id,
                    'mode': document_mode,
                    'timestamp': str(np.datetime64('now'))
                }
                
        except Exception as e:
            st.error(f"❌ Error fetching document: {str(e)}")
            st.session_state.current_document = None

with col2:
    if st.button("Clear Results"):
        st.session_state.current_document = None
        st.rerun()

# Display results if available
if hasattr(st.session_state, 'current_document') and st.session_state.current_document:
    doc_info = st.session_state.current_document
    document_data = doc_info['data']
    
    st.divider()
    
    # Document header
    st.subheader(f"{doc_info['mode']}: {doc_info['id']}")
    
    # Document metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Document ID", doc_info['id'])
    with col2:
        st.metric("Token Count", len(document_data))
    with col3:
        st.metric("Data Type", str(document_data.dtype))
    
    # Raw tokens display
    if show_raw_tokens:
        st.subheader("Raw Tokens")
        with st.expander("View Raw Token Array", expanded=False):
            st.code(str(document_data.tolist()[:100]) + ("..." if len(document_data) > 100 else ""))
            
            # Show first and last few tokens
            if len(document_data) > 20:
                st.write("**First 10 tokens:**")
                st.code(str(document_data[:10].tolist()))
                st.write("**Last 10 tokens:**")
                st.code(str(document_data[-10:].tolist()))
    
    # Detokenized text display
    if return_detokenized:
        st.subheader("Detokenized Text")
        try:
            detokenized_text = st.session_state.tokenizer.decode(document_data)
            
            # Display text with formatting
            st.text(
                detokenized_text,
                help="The document content converted back to readable text",
            )
            
            # Text statistics
            st.subheader("Text Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Character Count", len(detokenized_text))
            with col2:
                st.metric("Word Count", len(detokenized_text.split()))
            with col3:
                st.metric("Line Count", len(detokenized_text.split('\n')))
                
        except Exception as e:
            st.error(f"❌ Error during detokenization: {str(e)}")
    
    # Download options
    st.subheader("Export Options")
    col1, col2 = st.columns(2)
    
    with col1:
        # Download raw tokens as JSON
        tokens_json = json.dumps({
            'document_id': doc_info['id'],
            'mode': doc_info['mode'],
            'tokens': document_data.tolist(),
            'metadata': {
                'token_count': len(document_data),
                'dtype': str(document_data.dtype),
                'timestamp': doc_info['timestamp']
            }
        }, indent=2)
        
        st.download_button(
            label="Download Tokens (JSON)",
            data=tokens_json,
            file_name=f"document_{doc_info['id']}_{doc_info['mode'].lower().replace(' ', '_')}_tokens.json",
            mime="application/json"
        )
    
    with col2:
        # Download detokenized text if available
        if return_detokenized and 'detokenized_text' in locals():
            st.download_button(
                label="Download Text (TXT)",
                data=detokenized_text,
                file_name=f"document_{doc_info['id']}_{doc_info['mode'].lower().replace(' ', '_')}_text.txt",
                mime="text/plain"
            )

# Help section
with st.expander("ℹ️ Help", expanded=False):
    st.markdown("""
    ### Document Viewer Help
    
    **Document Modes:**
    - **Training Order**: View documents in the order they will be presented during training using `get_train_document_by_id`
    - **Corpus Order**: View documents in the order they are present in the corpus using `get_corpus_document_by_id`

    **Options:**
    - **Return detokenized text**: Convert tokens back to readable text (requires tokenizer)
    - **Show raw tokens**: Display the raw token array
    
    **Usage:**
    1. Select the document ordering (Training or Corpus)
    2. Enter the document ID
    3. Configure display options
    4. Click "View Document" to retrieve and display the document
    
    **Export:**
    - Download tokens as JSON format
    - Download detokenized text as TXT format (if tokenizer is available)
    """)
