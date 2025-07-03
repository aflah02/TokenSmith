import streamlit as st
import json
import numpy as np

st.title("Inspect Dataset")

# Check if inspect handler is initialized
if not hasattr(st.session_state, 'dataset_manager') or st.session_state.dataset_manager.inspect is None:
    st.error("❌ Inspect functionality not initialized. Please restart the app with proper CLI arguments.")
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

# List of function names to choose from
function_names = [
    "get_sequence",
    "get_batch",
]

st.sidebar.title("Select Function")
selected_function = st.sidebar.radio("Choose an operation:", function_names)

# Options for detokenization and document details
return_doc_details = st.sidebar.checkbox("Include document details", value=False)
return_detokenized = st.sidebar.checkbox("Return detokenized text", value=False)

if return_detokenized and not st.session_state.tokenizer:
    st.sidebar.warning("Tokenizer not available. Please provide tokenizer path in CLI args or disable detokenization.")
    return_detokenized = False

if selected_function == "get_sequence":
    sequence_id = st.number_input("Enter sequence ID:", min_value=0, value=0, step=1)
    
    if st.button("Get Sequence"):
        try:
            with st.spinner("Fetching sequence..."):
                result = st.session_state.dataset_manager.inspect.inspect_sample_by_id(
                    sample_id=sequence_id,
                    return_doc_details=return_doc_details,
                    return_detokenized=return_detokenized,
                    tokenizer=st.session_state.tokenizer if return_detokenized else None
                )
                
                if return_doc_details:
                    sequence_data, doc_details = result
                    
                    st.subheader("Sequence Data")
                    if return_detokenized:
                        st.text_area("Detokenized Text:", value=sequence_data, height=200, disabled=True)
                    else:
                        # Display tokens as arrays
                        st.write("Token Arrays:")
                        for i, arr in enumerate(sequence_data):
                            st.write(f"Array {i}: {arr.tolist()}")
                    
                    st.subheader("Document Details")
                    st.json(doc_details)
                    
                else:
                    st.subheader("Sequence Data")
                    if return_detokenized:
                        st.text_area("Detokenized Text:", value=result, height=200, disabled=True)
                    else:
                        # Display tokens as arrays
                        st.write("Token Arrays:")
                        for i, arr in enumerate(result):
                            st.write(f"Array {i}: {arr.tolist()}")
                            
        except Exception as e:
            st.error(f"Error fetching sequence: {e}")
            st.exception(e)

elif selected_function == "get_batch":
    batch_id = st.number_input("Enter batch ID:", min_value=0, value=0, step=1)
    batch_size = st.number_input("Enter batch size:", value=16, min_value=1, step=1)
    
    if st.button("Get Batch"):
        try:
            with st.spinner("Fetching batch..."):
                result = st.session_state.dataset_manager.inspect.inspect_sample_by_batch(
                    batch_id=batch_id,
                    batch_size=batch_size,
                    return_doc_details=return_doc_details,
                    return_detokenized=return_detokenized,
                    tokenizer=st.session_state.tokenizer if return_detokenized else None
                )
                
                st.subheader(f"Batch {batch_id} (Size: {batch_size})")
                
                for i, sample_result in enumerate(result):
                    with st.expander(f"Sample {i+1}"):
                        if return_doc_details:
                            if isinstance(sample_result, tuple):
                                sample_data, doc_details = sample_result
                                
                                st.write("**Sample Data:**")
                                if return_detokenized:
                                    st.text_area(f"Detokenized Text {i+1}:", value=sample_data, height=100, key=f"sample_{i}", disabled=True)
                                else:
                                    # Display tokens as arrays
                                    for j, arr in enumerate(sample_data):
                                        st.write(f"Array {j}: {arr.tolist()}")
                                
                                st.write("**Document Details:**")
                                st.json(doc_details)
                            else:
                                st.error(f"Unexpected result format for sample {i+1}")
                        else:
                            if return_detokenized:
                                st.text_area(f"Detokenized Text {i+1}:", value=sample_result, height=100, key=f"sample_{i}", disabled=True)
                            else:
                                # Display tokens as arrays
                                for j, arr in enumerate(sample_result):
                                    st.write(f"Array {j}: {arr.tolist()}")
                                    
        except Exception as e:
            st.error(f"Error fetching batch: {e}")
            st.exception(e)
