import streamlit as st
import random
import numpy as np
import pandas as pd
import altair as alt
import time

# List of function names to choose from
function_names = [
    "count",
    "contains",
    "positions",
    "count_next",
    # "batch_count_next",
    # "sample_smoothed",
    # "sample_unsmoothed",
    # "get_smoothed_probs",
    # "batch_get_smoothed_probs",
    # "estimate_delta"
]

st.sidebar.title("Select Function")
selected_function = st.sidebar.radio("Choose an operation:", function_names)

# Slider appears only for count_next
if selected_function == "count_next":
    show_top_k = st.sidebar.slider("Show top k results:", 1, 100, 10)
    normalize = st.sidebar.checkbox("Normalize distribution", value=False)

st.title("Search Dataset")

# Track last selected function
if "last_function" not in st.session_state:
    st.session_state.last_function = selected_function

# If function changed, reset query
if selected_function != st.session_state.last_function:
    st.session_state.query = ""
    st.session_state.last_function = selected_function

# enter query
query = st.text_input(
    "Enter your query:",
    value=st.session_state.get("query", ""),
    key="query"
)

if query:
    # Some logic goes here
    with st.spinner("Processing your query..."):
        time.sleep(2)
    # st.write("Output will be displayed here.")
    if selected_function == "count":
        st.write("Total count of tokens", 156)
    elif selected_function == "contains":
        st.write("Contains tokens", True)
    elif selected_function == "positions":
        st.write("Positions of tokens", [90, 678, 2485])
    elif selected_function == "count_next":
        # render a distribution over 50K tokens normalized to 1
        tokens = [i for i in range(1, 50001)]
        # random_distribution = {token: random.randint(1, 100) for token in tokens}
        random_distribution = {
            token: np.random.exponential(scale=3.0) for token in tokens
        }
        total = sum(random_distribution.values())
        if normalize:
            normalized_distribution = {
                token: value / total for token, value in random_distribution.items()
            }
        else:
            normalized_distribution = random_distribution
        # sort the distribution and take the top k
        sorted_distribution = sorted(
            normalized_distribution.items(), key=lambda item: item[1], reverse=True
        )
        # sorted_distribution already contains the top_k sorted list
        top_k = sorted_distribution[:show_top_k]

        # Create DataFrame
        df = pd.DataFrame(sorted_distribution[:show_top_k], columns=["Token", "Probability"])

        # Sort to ensure visual order
        df.sort_values(by="Probability", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)  # Add this line to reset index

        st.write(df)

        st.altair_chart(
            alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("Token:O", title="Token", sort=None),
                    y=alt.Y("Probability:Q", title="Probability" if normalize else "Count"),
                    tooltip=["Token", "Probability"]
                )
                .properties(width=700, height=400),
            use_container_width=True
        )