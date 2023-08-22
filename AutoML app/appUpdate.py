import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly_express as px
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder,LabelEncoder
from sklearn.impute import SimpleImputer





if os.path.exists('dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar:

    st.image("https://h2o.ai/platform/h2o-automl/_jcr_content/root/container/section_1366565710/par/advancedcolumncontro/columns0/image.coreimg.png/1678211341158/h2o-automl.png")
    st.title("AutoML project")
    st.info("This project is used to automat the process of EDA and ML modeling just upload your data and wait for the results")
    choice = st.radio("Navigation", ["Upload your data","Perform EDA",'Data Preparing',"Perform modeling", "Download"])


if choice == "Upload your data":
    st.title("Upload Your Dataset please!")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)


if choice == "Perform EDA":
    st.title("Exploratory Data Analysis")

    eda_choise = st.selectbox('Pick the operation you want',['Show shape','Summary','Show columns','Show selected columns',
                                                             'Show Value Counts','Correlation Plot','Pie Plot','Scatter Plot'])
    if eda_choise =='Show shape' :
        st.write(df.shape)

    if eda_choise =='Summary' :
        st.write(df.describe())

    if eda_choise =='Show columns' :
        all_columns = df.columns
        st.write(all_columns)

    if eda_choise =='Show selected columns' :
        selected_columns = st.multiselect('Select desired columns',df.columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)

    if eda_choise =='Show Value Counts' :
        try:
            selected_columns = st.multiselect('Select desired columns', df.columns)
            new_df = df[selected_columns]
            st.write(new_df.value_counts().rename(index='Value'))
        except:
            pass

    if eda_choise =='Correlation Plot' :
        plt.figure(figsize=(10, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        st.pyplot(plt)

    if eda_choise =='Pie Plot' :
        column_to_plot = st.selectbox("Select 1 Column", df.columns)
        value_counts = df[column_to_plot].value_counts()
        fig, ax = plt.subplots()
        ax.pie(value_counts, labels=value_counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis('equal')
        st.write(fig)

    if eda_choise =='Scatter Plot' :
        try :
            selected_columns = st.multiselect('Select two columns',df.columns)
            first_column = selected_columns[0]
            second_column = selected_columns[1]
            fig = px.scatter(df, x=first_column, y=second_column)
            fig.update_layout(title="Scatter Plot", xaxis_title=first_column, yaxis_title=second_column)
            st.plotly_chart(fig)
        except:
            pass


if choice == "Data Preparing" :

    st.title('Preparing the data before machine learning modeling')


    want_to_drop = st.selectbox('Do you want to drop any columns ?',['','Yes','No'])
    st.warning('It is recommended to drop columns such as name customer ID ,etc ')
    if want_to_drop == 'No' :
        st.write('OK, Please processed to next step')
    if want_to_drop == 'Yes' :
        columns_to_drop = st.multiselect('Select desired columns to drop', df.columns)
        df = df.drop(columns_to_drop,axis=1)
        st.dataframe(df)

    encoder_option = st.selectbox('Do you want to encode your data ?',['','Yes','No'])
    if encoder_option == 'No' :
        st.write('OK, Please processed to next step')

    if encoder_option == 'Yes' :
        encoder_columns = st.multiselect('Please pick the columns you want to encode',df.columns)
        encoder_type = st.selectbox('Please pick the type of encoder you want to use', ['Label Encoder','One Hot Encoder'])
        if encoder_type == 'Label Encoder' :
            encoder = LabelEncoder()
            df[encoder_columns] = df[encoder_columns].apply(encoder.fit_transform)
            st.dataframe(df)
        if encoder_type == 'One Hot Encoder':
            df = pd.get_dummies(df, columns=encoder_columns, prefix=encoder_columns,drop_first=True)
            st.dataframe(df)


    fill_option = st.selectbox('Is there any missing data you want to fill ?', ['', 'Yes', 'No'])

    if fill_option == 'No':
        st.write('OK, Please processed to next step')

    if fill_option == 'Yes':
        encoder_columns = st.multiselect('Please pick the columns you want to fill', df.columns)
        encoder_type = st.selectbox('Please pick the type of filling you want to use', ['Mean','Median','Most frequent'])
        try:

            if encoder_type == 'Mean' :
                imputer = SimpleImputer(strategy='mean')
                df[encoder_columns] = np.round(imputer.fit_transform(df[encoder_columns]),1)
                st.dataframe(df)

            if encoder_type == 'Median' :
                imputer = SimpleImputer(strategy='median')
                df[encoder_columns] = np.round(imputer.fit_transform(df[encoder_columns]),1)
                st.dataframe(df)

            if encoder_type == 'Most frequent' :
                imputer = SimpleImputer(strategy='most_frequent')
                df[encoder_columns] = np.round(imputer.fit_transform(df[encoder_columns]),1)
                st.dataframe(df)

        except :
            pass

df = df.to_csv('dataset.csv', index=None)

if choice == "Perform modeling":

    st.title('It is time for Machine Learning modeling')
    df = pd.read_csv('dataset.csv', index_col=None)

    target = st.selectbox('Choose your target variable', df.columns)
    X = df.drop(columns=target)
    y = df[target]
    st.write('Your Features are', X)
    st.write('Your Target is', y)

    modeling_choise = st.selectbox('Do you want Auto modeling or you want to choose the model ?',['Auto modeling','Choose model'])



    test_size = st.select_slider('Pick the test size you want', range(1, 100, 1))
    st.warning('It is recommended to pick a number between 10 and 30 ')
    test_size_fraction = test_size / 100.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_fraction, random_state=42)
    st.write('Shape of training data is :',X_train.shape)
    st.write('Shape of testing data is :',X_test.shape)

    task_type = st.selectbox('Choose type of task you want to apply',['Classification','Regression'])













