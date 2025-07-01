import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import time
import json

# List of function names to choose from
function_names = [
    "count",
    "contains",
    "positions",
    "count_next"
]

st.sidebar.title("Select Function")
selected_function = st.sidebar.radio("Choose an operation:", function_names)

# Function-specific parameters
if selected_function == "count_next":
    show_top_k = st.sidebar.slider("Show top k results:", 1, 100, 10)
    normalize = st.sidebar.checkbox("Normalize distribution", value=False)

st.title("Search Dataset")

# Check if search handler is initialized
if not hasattr(st.session_state, 'dataset_manager') or st.session_state.dataset_manager.search is None:
    st.error("❌ Search functionality not initialized. Please restart the app with proper CLI arguments.")
    st.write("Required arguments:")
    st.code("""
streamlit run app.py -- \\
    --bin-file-path /path/to/data.bin \\
    --search-index-path /path/to/search.idx \\
    --vocab 65536
    """)
    st.stop()

# Track last selected function
if "last_function" not in st.session_state:
    st.session_state.last_function = selected_function

# If function changed, reset query
if selected_function != st.session_state.last_function:
    st.session_state.query = ""
    st.session_state.last_function = selected_function

# Query input handling
query_input = st.text_input(
    "Enter your query (as JSON array of token IDs):",
    value=st.session_state.get("query", ""),
    key="query",
    help="Example: [101, 2023, 102]"
)

query = []
if query_input.strip():
    try:
        query = json.loads(query_input.strip())
        if not isinstance(query, list) or not all(isinstance(x, int) for x in query):
            st.error("Query must be a list of integers")
            st.stop()
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON format: {e}")
        st.stop()

# Execute the selected function
if query:
    try:
        with st.spinner("Processing your query..."):
            search_handler = st.session_state.dataset_manager.search
            
            if selected_function == "count":
                result = search_handler.count(query)
                st.metric("Total count of tokens", result)
                
            elif selected_function == "contains":
                result = search_handler.contains(query)
                st.write(f"Contains tokens: {result}")
                
            elif selected_function == "positions":
                result = search_handler.positions(query)
                st.write(f"Positions of tokens: {result}")
                if result:
                    df = pd.DataFrame({"Position": result})
                    st.dataframe(df)
                    
            elif selected_function == "count_next":
                result = search_handler.count_next(query)
                if result:
                    # Convert to DataFrame for display
                    counts_dict = {i: count for i, count in enumerate(result) if count > 0}
                    if counts_dict:
                        sorted_items = sorted(counts_dict.items(), key=lambda x: x[1], reverse=True)
                        top_k_items = sorted_items[:show_top_k]
                        
                        if normalize:
                            total = sum(counts_dict.values())
                            top_k_items = [(token, count/total) for token, count in top_k_items]
                        
                        df = pd.DataFrame(top_k_items, columns=["Token", "Probability" if normalize else "Count"])
                        st.dataframe(df)
                        
                        # Create visualization
                        chart = alt.Chart(df).mark_bar().encode(
                            x=alt.X("Token:O", title="Token"),
                            y=alt.Y("Probability:Q" if normalize else "Count:Q", 
                                   title="Probability" if normalize else "Count"),
                            tooltip=["Token", "Probability" if normalize else "Count"]
                        ).properties(width=700, height=400)
                        
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.write("No results found.")
                else:
                    st.write("No results found.")
                    
    except Exception as e:
        st.error(f"Error executing {selected_function}: {e}")
        st.exception(e)
