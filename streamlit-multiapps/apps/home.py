import streamlit as st
import pandas as pd
import numpy as np


def app():
    # with open('static/style.css','r') as f:
    #     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    st.write("# MACHINE LEARNING PROJECT")
    st.write("## 1. Topics and Members :male-scientist: ")
    st.write("### Members")
    st.write("1. Phạm Tiến Dũng - 20110622")
    st.write("2. Phạm Phúc Bình - 20110252")
    st.write("### Topics")
    st.write("1. Poker card detection")
    st.write("2. Weather prediction")
    st.write("## 2. How to use :arrows_counterclockwise: :+1: :white_check_mark: ")
    st.write("### Install environment:")
    st.write("* `cd ./streamlit-multiapps`")
    st.write("* `pip install -r requirements.txt`")
    st.write("### Run application:")
    st.write("1. `streamlit run app.py`")
