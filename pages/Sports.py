import streamlit as st
import json
from pages.function.predictionapi import post, plot_result, post_azure, plot_object
import pandas as pd


with st.sidebar:
    values = st.slider(
        '顯示多少個分類項目:',
        1, 100, 5)
    uploaded_file = st.file_uploader("選擇檔案", type=['png','jpg','jpeg'])

st.header('分類視覺圖')
if uploaded_file is not None:
    results = post(uploaded_file, n=values)
    fig = plot_result(uploaded_file, results)
    st.pyplot(fig)

st.header('分類資料')
if uploaded_file is not None:
    results = post(uploaded_file, n=values)
    prediction = {results[result]['label']:results[result]['prob'] for result in results}
    df = pd.DataFrame.from_dict(prediction, orient='index')
    df.columns = ['Probability']
    st.write(df)

st.header('Azure Service')
if uploaded_file is not None:
    results = post_azure(uploaded_file)
    df = pd.DataFrame.from_dict(results, orient='index')
    df.columns = ['Info']
    st.write(df)
    im = plot_object(uploaded_file, results)
    st.image(im)
