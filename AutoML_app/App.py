# App Link : https://electro-pi-automl.streamlit.app/


import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly_express as px
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.impute import SimpleImputer
import pickle


if os.path.exists('dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar:

    st.image("https://h2o.ai/platform/h2o-automl/_jcr_content/root/container/section_1366565710/par/advancedcolumncontro/columns0/image.coreimg.png/1678211341158/h2o-automl.png")
    st.title("AutoML project")
    st.info("This project powered by electropi.ai Upload your data and choose type your EDA and Prepare your data for ML modeling")
    choice = st.radio("Choose the Desired operation", ["Upload your data","Perform EDA",'Data Preparing and Modeling'])


if choice == "Upload your data":
    st.title("Upload Your Dataset please!")
    file = st.file_uploader("Upload Your Dataset")

    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)


if choice == "Perform EDA":
    st.title("Exploratory Data Analysis")

    eda_choise = st.selectbox('Pick the operation you want',['','Show shape','Show data type','Show messing values','Summary',
                                                             'Show columns','Show selected columns','Show Value Counts'])
    if eda_choise =='Show shape' :
        st.write(df.shape)

    if eda_choise =='Show data type' :
        st.write(df.dtypes)

    if eda_choise == 'Show messing values' :
        st.write(df.isna().sum())

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


    plot_choice = st.selectbox('Select type of plot you want :',['','Box Plot','Correlation Plot','Pie Plot',
                                                                 'Scatter Plot','Bar Plot'])


    if plot_choice == 'Box Plot' :
        column_to_plot = st.selectbox("Select 1 Column", df.columns)
        fig = px.box(df,y=column_to_plot)
        st.plotly_chart(fig)

    if plot_choice =='Correlation Plot' :
        plt.figure(figsize=(10, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        st.pyplot(plt)

    if plot_choice =='Pie Plot' :
        column_to_plot = st.selectbox("Select 1 Column", df.columns)
        value_counts = df[column_to_plot].value_counts()
        fig, ax = plt.subplots()
        ax.pie(value_counts, labels=value_counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis('equal')
        st.write(fig)

    if plot_choice =='Scatter Plot' :

        try :

            selected_columns = st.multiselect('Select two columns',df.columns)
            first_column = selected_columns[0]
            second_column = selected_columns[1]
            fig = px.scatter(df, x=first_column, y=second_column)
            fig.update_layout(title="Scatter Plot", xaxis_title=first_column, yaxis_title=second_column)
            st.plotly_chart(fig)

        except:
            pass

    if plot_choice == 'Bar Plot':

        try :

            selected_columns = st.multiselect('Select columns', df.columns)
            first_column = selected_columns[0]
            second_column = selected_columns[1]

            fig = px.bar(df, x=first_column, y=second_column, title='Bar Plot')
            st.plotly_chart(fig)

        except :
            pass



if choice == "Data Preparing and Modeling" :

    st.title('Preparing the data before machine learning modeling')


    want_to_drop = st.selectbox('Do you want to drop any columns ?',['','Yes','No'])

    if want_to_drop == 'No':

        st.warning('It is recommended to drop columns such as name, customer ID, etc.')

    if want_to_drop == 'Yes':

        columns_to_drop = st.multiselect('Select columns to drop', df.columns)
        if columns_to_drop  :
            df = df.drop(columns_to_drop, axis=1)
            st.success('Columns dropped successfully.')
            st.dataframe(df)

    target_choices = [''] + df.columns.tolist()
    target = st.selectbox('Choose your target variable', target_choices)

    categorical_columns = df.select_dtypes(include=['object', 'category'])
    categorical_columns = [col for col in categorical_columns if col != target]

    one_hot_encoded = pd.get_dummies(categorical_columns, drop_first=True)
    df_encoded = pd.concat([df, one_hot_encoded], axis=1)
    columns_to_drop = categorical_columns
    df_encoded = df_encoded.drop(columns=columns_to_drop)

    df_filled = df_encoded.fillna(df_encoded.median())

    from sklearn.preprocessing import MinMaxScaler
    try:
        X = df_filled.drop(columns=target)
        y = df_filled[target]
        st.write('Your Features are', X)
        st.write('Your Target is', y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        mm = MinMaxScaler()
        scaled_data = mm.fit_transform(X)
        df_scaled = pd.DataFrame(scaled_data, columns=X.columns)
        df_scaled['target'] = y
        st.write('your data after preprocessing', df_scaled)

    except:
        pass



    try :
        if y.dtype == 'object' or y.nunique() <= 10:
            st.info('This is a classification problem')
            modeling_choice = st.selectbox('Do you want Auto modeling or you want to choose the model ?',
                                           ['', 'Auto modeling', 'Manual modeling'])

            if modeling_choice == 'Auto modeling':
                from pycaret.classification import *

                if st.button('Run Modelling'):

                    setup(df, target=target, verbose=False)
                    setup_df = pull()
                    st.info("This is the ML experiment settings")
                    st.dataframe(setup_df)
                    st.error('IT WILL TAKE TWO MINUTES PLEASE BE PATIENT')
                    best_model = compare_models(include=['lr','dt','rf'])
                    compare_df = pull()
                    st.info("This is your ML model")
                    st.dataframe(compare_df)
                    save_model(best_model, 'best_model')

                    with open('best_model.pkl', 'rb') as model_file:
                        st.download_button('Download the model', model_file, 'best_model.pkl')

            if modeling_choice == 'Manual modeling':

                algo_type = st.selectbox('Please choose which type of algorithm you want to use',
                                         ['', 'Logistic Regression', 'Decision Trees', 'Random Forest', 'SVC',
                                          'KNN'])

                if algo_type == 'Logistic Regression':
                    from sklearn.linear_model import LogisticRegression

                    clf = LogisticRegression(random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                if algo_type == 'Decision Trees':
                    from sklearn.tree import DecisionTreeClassifier

                    clf = DecisionTreeClassifier(random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                if algo_type == 'Random Forest':
                    from sklearn.ensemble import RandomForestClassifier

                    clf = RandomForestClassifier(random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                if algo_type == 'SVC':
                    from sklearn.svm import SVC

                    clf = SVC(random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                if algo_type == 'KNN':
                    from sklearn.neighbors import KNeighborsClassifier

                    clf = KNeighborsClassifier()
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                evaluation_type = st.selectbox('Choose type of evaluation metrics ', ['', 'Accuracy', 'Confusion Matrix',
                                                                                      'Precision, Recall, and F1-score'])

                if evaluation_type == 'Accuracy':
                    from sklearn.metrics import accuracy_score

                    accuracy = accuracy_score(y_test, y_pred)
                    st.write("Accuracy:", accuracy)

                if evaluation_type == 'Confusion Matrix':
                    from sklearn.metrics import confusion_matrix

                    cm = confusion_matrix(y_test, y_pred)
                    st.write("Confusion Matrix:")
                    st.dataframe(cm)

                if evaluation_type == 'Precision, Recall, and F1-score':
                    from sklearn.metrics import precision_score, recall_score, f1_score

                    precision = precision_score(y_test, y_pred, average='macro')
                    recall = recall_score(y_test, y_pred, average='macro')
                    f1 = f1_score(y_test, y_pred, average='macro')
                    metrics_dict = {
                        "Metric": ["Precision", "Recall", "F1-Score"],
                        "Value": [precision, recall, f1]
                    }
                    metrics_df = pd.DataFrame(metrics_dict)
                    st.dataframe(metrics_df)

                try:

                    model_filename = "clf.pkl"
                    with open(model_filename, "wb") as model_file:
                        pickle.dump(clf, model_file)

                    st.download_button('Download the model', open(model_filename, 'rb').read(), 'clf.pkl')

                except:
                    pass



        else:
            st.info('This is a regression problem')
            modeling_choice = st.selectbox('Do you want Auto modeling or you want to choose the model ?',
                                           ['', 'Auto modeling', 'Manual modeling'])

            if modeling_choice == 'Auto modeling':

                from pycaret.regression import *

                if st.button('Run Modelling'):
                    setup(df, target=target, verbose=False)
                    setup_df = pull()
                    st.info("This is the ML experiment settings")
                    st.dataframe(setup_df)
                    st.error('IT WILL TAKE TWO MINUTES PLEASE BE PATIENT')
                    best_model = compare_models(include=['lr','ridge','lasso'])
                    compare_df = pull()
                    st.info("This is your ML model")
                    st.dataframe(compare_df)
                    save_model(best_model, 'best_model')

                    with open('best_model.pkl', 'rb') as model_file:
                        st.download_button('Download the model', model_file, 'best_model.pkl')

            if modeling_choice == 'Manual modeling':

                algo_type = st.selectbox('Please choose which type of algorithm you want to use',
                                         ['', 'Linear Regression', 'Ridge', 'SVR', 'Random Forest'])

                if algo_type == 'Linear Regression':
                    from sklearn.linear_model import LinearRegression

                    rg = LinearRegression()
                    rg.fit(X_train, y_train)
                    y_pred = rg.predict(X_test)

                if algo_type == 'Ridge':
                    from sklearn.linear_model import Ridge

                    rg = Ridge()
                    rg.fit(X_train, y_train)
                    y_pred = rg.predict(X_test)

                if algo_type == 'SVR':
                    from sklearn.svm import SVR

                    rg = SVR()
                    rg.fit(X_train, y_train)
                    y_pred = rg.predict(X_test)

                if algo_type == 'Random Forest':
                    from sklearn.ensemble import RandomForestRegressor

                    rg = RandomForestRegressor()
                    rg.fit(X_train, y_train)
                    y_pred = rg.predict(X_test)

                evaluation_type = st.selectbox('Choose type of evaluation metrics ', ['', 'MAE', 'MSE', 'r2 score'])

                if evaluation_type == 'MAE':
                    from sklearn.metrics import mean_absolute_error

                    MAE = mean_absolute_error(y_test, y_pred)
                    st.write("Mean absolute error:", MAE)

                if evaluation_type == 'MSE':
                    from sklearn.metrics import mean_squared_error

                    MSE = mean_squared_error(y_test, y_pred)
                    st.write("Mean squared error:", MSE)

                if evaluation_type == 'r2 score':
                    from sklearn.metrics import r2_score

                    r2 = r2_score(y_test, y_pred)
                    st.write("r2 score:", r2)

                try:

                    model_filename = "rg.pkl"
                    with open(model_filename, "wb") as model_file:
                        pickle.dump(rg, model_file)

                    st.download_button('Download the model', open(model_filename, 'rb').read(), 'rg.pkl')

                except:
                    pass

    except :
        pass
















