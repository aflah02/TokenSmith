import streamlit as st

search_page = st.Page("search.py", title="Search Dataset", icon=":material/find_in_page:")
inspect_page = st.Page("inspect.py", title="Inspect Dataset", icon=":material/eye_tracking:")

pg = st.navigation([search_page, inspect_page])
st.set_page_config(page_title="TokenSmith UI", page_icon=":material/key:", layout="wide")
pg.run()