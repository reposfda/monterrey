import pandas as pd
import streamlit as st

def render_mty_table(df: pd.DataFrame, *, index: bool = False):
    st.markdown(df.to_html(index=index, classes="mty-table"), unsafe_allow_html=True)