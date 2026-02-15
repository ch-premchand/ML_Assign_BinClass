import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(page_title="ML Classification Models", layout="wide")

st.title("ML Classification Models Comparison")
st.markdown("Upload your dataset and compare 6 different classification models!")


st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Features", df.shape[1] - 1)

    st.subheader("Data Preview")
    st.dataframe(df.head(10))

    if df.shape[0] <500:
        st.warning("The dataset has less than 500 rows.")
    if df.shape[1] < 13:
        st.warning("The dataset has less than 13 columns.")



else:
    st.info("Please upload a CSV file !")

