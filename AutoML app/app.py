import pandas as pd
import streamlit as st
from ydata_profiling import profile_report
from streamlit_pandas_profiling import st_profile_report
import os
from pycaret.regression import *
from pycaret.classification import *
if os.path.exists('dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://h2o.ai/platform/h2o-automl/_jcr_content/root/container/section_1366565710/par/advancedcolumncontro/columns0/image.coreimg.png/1678211341158/h2o-automl.png")
    st.title("AutoML project")
    st.info("This project is used to automat the process of EDA and ML modeling just upload your data and wait for the results")
    choice = st.radio("Navigation", ["Upload your data","Perform EDA","Perform modeling", "Download"])

if choice == "Upload your data":
    st.title("Upload Your Dataset and see the magic!")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Perform EDA": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Perform modeling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'):
        setup(df, target=chosen_target, verbose=False)
        setup_df = pull()
        st.info("This is the ML experiment settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is your ML model")
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download": 
        with open('best_model.pkl','rb') as f:
            st.download_button('Download the model',f,'best_model.pkl')