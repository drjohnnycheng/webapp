import streamlit as st
import pandas as pd
from io import StringIO

uploaded_file = st.file_uploader("Upload a text file", type=["txt", "csv"])
if uploaded_file is not None:
    st.write(f"File: {uploaded_file.name}")

    # To read file as bytes:
    # bytes_data = uploaded_file.getvalue()
    # st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write("[Text File]")
    st.write(stringio)

    # To read file as string:
    # string_data = stringio.read()
    # st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    st.write("[CSV File]")
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)