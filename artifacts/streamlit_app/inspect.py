import streamlit as st

st.title("Inspect Dataset")

# List of function names to choose from
function_names = [
    "get_sequence",
    "get_batch",
]

st.sidebar.title("Select Function")
selected_function = st.sidebar.radio("Choose an operation:", function_names)

if selected_function == "get_sequence":
    sequence_id = st.text_input("Enter sequence number:", "")
    if sequence_id:
        st.write(f"Fetching sequence for number: {sequence_id}")
        # Dummy output
        dummy_sequence = {
            "sequence_id": sequence_id,
            "tokens": [101, 2023, 2003, 1037, 3978, 7099, 102],
            "text": "This is a dummy output."
        }
        st.json(dummy_sequence)

elif selected_function == "get_batch":
    batch_id = st.text_input("Enter batch number:", "")
    batch_size = st.number_input("Enter batch size:", value=16, min_value=1, step=1)
    if batch_id:
        st.write(f"Fetching batch for number: {batch_id} with size: {batch_size}")
        # Dummy output
        dummy_batch = [
            {
                "sequence_id": f"{batch_id}_{i}",
                "tokens": [101, 1045, 2293, 2651, 999, 102],
                "text": "I love pizza!"
            }
            for i in range(batch_size)
        ]
        st.json(dummy_batch)
