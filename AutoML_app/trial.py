import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn import datasets
import pycaret.classification as pyc_cls
import pycaret.regression as pyc_reg

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    # Add custom CSS style to position the button at the top-left corner
    st.markdown(
        """
        <style>
        .custom-button {
            position: absolute;
            top: 10px;
            left: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Use st.empty() to create an empty container at the top-left corner
    button_container = st.empty()

    # Button to be placed at the top-left corner
    if button_container.button("**Contact us**", key="costom_button"):
        st.write("khadidja_mek@hotmail.fr")

with col2:
    # Add custom CSS style to position the button at the top-left and in the middle
    st.markdown(
        """
        <style>
        .custom-button {
            position: absolute;
            top: 50%;
            left: 10px;
            transform: translateY(-50%);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Use st.empty() to create an empty container at the top-left and in the middle
    button_container = st.empty()

    # Button to be placed at the top-left and in the middle
    if button_container.button("**Author**", key="my_custom"):
        st.write('Khadidja Mekiri')

with col3:
    # Add custom CSS style to position the button at the top-left corner
    st.markdown(
        """
        <style>
        .custom-button {
            position: absolute;
            top: 10px;
            left: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Use st.empty() to create an empty container at the top-left corner
    button_container = st.empty()

    if button_container.button("**Satisfaction**", key="custom"):
        st.selectbox("Rate your satisfaction (1-5)", range(1, 6))

with col4:
    # Add custom CSS style to position the button at the top-left corner
    st.markdown(
        """
        <style>
        .custom-button {
            position: absolute;
            top: 10px;
            left: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Use st.empty() to create an empty container at the top-left corner
    button_container = st.empty()
    if button_container.button("**Datasets**", key="Data"):
        st.write("https://www.kaggle.com/datasets/morriswongch/kaggle-datasets")

with col5:
    # Add custom CSS style to position the button at the top-left corner
    st.markdown(
        """
        <style>
        .custom-button {
            position: absolute;
            top: 10px;
            left: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Use st.empty() to create an empty container at the top-left corner
    button_container = st.empty()
    if button_container.button("**About us**", key="info"):
        st.write("VisualModel360 a complete modeling tool that can do visualisation and prediction")

st.sidebar.header('Help Menu')

# Add a button in the sidebar
show_steps = st.sidebar.button('Show Menu')

if show_steps:
    with st.sidebar:
        st.write(" 🌟Welcome to VisualModel360!🌟")
        st.write("Here's how to dive right into our powerful data visualization and modeling platform:")
        st.write(
            'Upload Your Data: Simply drag and drop your dataset, or select one from the predefined options to get started.')
        st.write('Define Your Task: Are you interested in classification or regression? Let us know!')
        st.write('Pick Your Model: Browse through our selection and pick the model that best fits your needs.')
        st.write('Sit Back & Relax: Once you are set up, watch VisualModel360 work its magic.')
        st.write('Happy analyzing and enjoy your data journey with us! 🚀')
# Use raw HTML to change the title color
st.markdown('<style>h1{color: blue;}</style>', unsafe_allow_html=True)

st.title("VisualModel360")


def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, skip_blank_lines=True)
            if df.empty:
                st.error("Uploaded file is empty or incorrectly formatted.")
                return None
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
            if df.empty:
                st.error("Uploaded file is empty or incorrectly formatted.")
                return None
        else:
            st.error("This file format is not supported.")
            return None

        return df

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None


def generate_report(df):
    # Generate report using Pandas Profiling
    profile = ProfileReport(df, explorative=True, minimal=True)
    st_profile_report(profile)  # This will display the report in the streamlit app


def load_ready_dataset(dataset_name):
    if dataset_name == 'iris':
        data = datasets.load_iris(as_frame=True)
    elif dataset_name == 'wine':
        data = datasets.load_wine(as_frame=True)
    elif dataset_name == 'digits':
        data = datasets.load_digits(as_frame=True)
    elif dataset_name == 'breast_cancer':
        data = datasets.load_breast_cancer(as_frame=True)
    elif dataset_name == 'boston':
        data = datasets.load_boston(as_frame=True)
    elif dataset_name == 'diabetes':
        data = datasets.load_diabetes(as_frame=True)
    else:
        data = None

    if data:
        df = data.data

        if dataset_name in ['boston', 'diabetes']:
            df['target_value'] = data.target
        else:
            df['target'] = data.target
        return df
    else:
        return None


def handle_missing_values(df):
    for column in df.select_dtypes(include=[np.number]):
        df[column] = df[column].fillna(df[column].mean())
    for column in df.select_dtypes(include=[np.object]):
        df[column] = df[column].fillna(df[column].mode()[0])
    return df


def preprocess_data(df):
    le = LabelEncoder()
    scaler = StandardScaler()
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        df[column] = df[column].astype(str)
        df[column] = le.fit_transform(df[column])
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for column in numerical_columns:
        df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
    return df


def visualize_data(df):
    if len(df.columns) > 1:
        fig, ax = plt.subplots()
        df[df.columns[:2]].plot(kind='scatter', x=df.columns[0], y=df.columns[1], ax=ax)
        st.pyplot(fig)

    fig, ax = plt.subplots()
    df[df.columns[0]].hist(ax=ax)
    st.pyplot(fig)

    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for column in numerical_columns:
        fig, ax = plt.subplots()
        df.boxplot(column=[column], ax=ax)
        st.pyplot(fig)

        fig, ax = plt.subplots()
        df[column].plot.line(ax=ax)
        st.pyplot(fig)

    if len(numerical_columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df[numerical_columns].corr()
        sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f", ax=ax)
        st.pyplot(fig)


def choose_model(task):
    if task == "classification":
        model_choice = st.selectbox("Choose a model for classification",
                                    ["Logistic Regression", "Random Forest Classifier",
                                     "Support Vector Machine (SVC)", "K-Nearest Neighbors (KNN)",
                                     "Decision Tree Classifier", "Gradient Boosting Classifier",
                                     "Neural Network (MLP Classifier)"])

        if model_choice == "Logistic Regression":
            return LogisticRegression()
        elif model_choice == "Random Forest Classifier":
            return RandomForestClassifier()
        elif model_choice == "Support Vector Machine (SVC)":
            return SVC()
        elif model_choice == "K-Nearest Neighbors (KNN)":
            return KNeighborsClassifier()
        elif model_choice == "Decision Tree Classifier":
            return DecisionTreeClassifier()
        elif model_choice == "Gradient Boosting Classifier":
            return GradientBoostingClassifier()
        elif model_choice == "Neural Network (MLP Classifier)":
            return MLPClassifier()

    else:
        model_choice = st.selectbox("Choose a model for regression",
                                    ["Linear Regression", "Random Forest Regressor",
                                     "Support Vector Regression (SVR)", "K-Nearest Neighbors (KNN Regression)",
                                     "Decision Tree Regressor", "Gradient Boosting Regressor",
                                     "Neural Network (MLP Regressor)"])

        if model_choice == "Linear Regression":
            return LinearRegression()
        elif model_choice == "Random Forest Regressor":
            return RandomForestRegressor()
        elif model_choice == "Support Vector Regression (SVR)":
            return SVR()
        elif model_choice == "K-Nearest Neighbors (KNN Regression)":
            return KNeighborsRegressor()
        elif model_choice == "Decision Tree Regressor":
            return DecisionTreeRegressor()
        elif model_choice == "Gradient Boosting Regressor":
            return GradientBoostingRegressor()
        elif model_choice == "Neural Network (MLP Regressor)":
            return MLPRegressor()


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Function to plot confusion matrix using seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False,
                ax=ax, xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.write(fig)


def compute_mape(y_true, y_pred):
    """Compute Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def is_target_continuous(df, target_column, threshold=10):
    """
    Function to determine if target column is continuous or categorical.
    Assumes target is continuous if number of unique values > threshold and is of numeric type.
    """
    if df[target_column].nunique() > threshold and np.issubdtype(df[target_column].dtype, np.number):
        return True
    return False


def main():
    dataset_source = st.selectbox("Choose your dataset source:",
                                  ["Select a source", "Upload CSV or Excel", "Use ready datasets"])
    df = None

    if dataset_source == "Upload CSV or Excel":
        uploaded_file = st.file_uploader("Upload your CSV or Excel file", ["csv", "xlsx"])
        if uploaded_file:
            df = load_data(uploaded_file)
            if df is not None:
                st.write("Original Data:")
                st.write(df.head())
            else:
                st.warning("Error loading data. Please try again.")

    elif dataset_source == "Use ready datasets":
        ready_dataset_choice = st.selectbox("Select a dataset:",
                                            ["iris", "wine", "digits", "breast_cancer", "boston", "diabetes"])
        df = load_ready_dataset(ready_dataset_choice)
        if df is not None:
            st.write(f"{ready_dataset_choice.replace('_', ' ').capitalize()} Dataset:")
            st.write(df.head())
        else:
            st.warning(f"Error loading the {ready_dataset_choice} dataset. Please select another.")

        # If a dataframe is loaded (either uploaded or from the ready datasets)
        # if df is not None:
        #     df = preprocess_data(df)
        #     options = st.multiselect("What would you like to perform?", ["Pandas Profiling", "Visualization", "Prediction"])

        #     if "Pandas Profiling" in options:
        #         st.write("Generating Pandas Profiling Report...")
        #         generate_report(df)

        #     if "Visualization" in options:
        #         visualize_data(df)

        #     if "Prediction" in options:
        #         target_column = st.selectbox('Select your target column:', df.columns)
        #         numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        #         target_continuous = target_column in numerical_columns

        #         if target_continuous:
        #             task = 'regression'
        #             st.write("Based on the continuous nature of the target column, Regression will be applied.")
        #         else:
        #             task = 'classification'
        #             st.write("Based on the categorical nature of the target column, Classification will be applied.")

        #         df[target_column] = df[target_column].astype(int)
        #         model = choose_model(task)
        #         X_train, X_test, y_train, y_test = train_test_split(df.drop(target_column, axis=1), df[target_column], test_size=0.2)
        #         model.fit(X_train, y_train)
        #         if task == 'classification':
        #             display_classification_results(X_test, y_test, model)
        #         else:
        #             display_regression_results(X_test, y_test, model)

        if df is not None:
            df = preprocess_data(df)
            options = st.multiselect("What would you like to perform?",
                                     ["Visualization", "Prediction", "PyCaret Multi-Algorithm Analysis"])

        if "Visualization" in options:
            visualize_data(df)

        if "Prediction" in options:
            target_column = st.selectbox('Select your target column:', df.columns)
            numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
            target_continuous = target_column in numerical_columns

            if target_continuous:
                task = 'regression'
                st.write("Based on the continuous nature of the target column, Regression will be applied.")
            else:
                task = 'classification'
                st.write("Based on the categorical nature of the target column, Classification will be applied.")

            df[target_column] = df[target_column].astype(int)
            model = choose_model(task)
            X_train, X_test, y_train, y_test = train_test_split(df.drop(target_column, axis=1), df[target_column],
                                                                test_size=0.2)
            model.fit(X_train, y_train)
            if task == 'classification':
                display_classification_results(X_test, y_test, model)
            else:
                display_regression_results(X_test, y_test, model)

        if "PyCaret Multi-Algorithm Analysis" in options:
            target_column = st.selectbox('Select your target column for PyCaret:', df.columns)
            numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
            target_continuous = target_column in numerical_columns

            if target_continuous:
                st.write("Using PyCaret's Regression Module...")
                pyc_setup = pyc_reg.setup(df, target=target_column, silent=True, verbose=False, session_id=123)
                pyc_comparison = pyc_reg.compare_models(verbose=False)
            else:
                st.write("Using PyCaret's Classification Module...")
                pyc_setup = pyc_cls.setup(df, target=target_column, silent=True, verbose=False, session_id=123)
                pyc_comparison = pyc_cls.compare_models(verbose=False)

            st.write(pyc_comparison)


def display_classification_results(X_test, y_test, model):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    st.write(f"Accuracy: {accuracy:.2f}%")
    confidence_level = accuracy * 100
    st.write('Confidence Level')

    if confidence_level >= 90:
        bar_color = 'green'
    elif confidence_level >= 70:
        bar_color = 'yellow'
    else:
        bar_color = 'red'

    progress_html = f"""
    <div style="position: relative; width: 100%; height: 25px; background-color: #f0f0f0; border-radius: 5px;">
        <div style="position: absolute; width: {confidence_level}%; height: 100%; background-color: {bar_color}; border-radius: 5px; transition: width 0.5s;"></div>
        <div style="position: absolute; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; font-weight: bold; color: black ;">
            {confidence_level:.2f}%
        </div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)
    st.write("Classification Report:")
    st.text(classification_report(y_test, predictions))
    # Plotting confusion matrix
    class_names = list(set(y_test))  # Assuming target column has string labels
    plot_confusion_matrix(y_test, predictions, class_names)


def display_regression_results(X_test, y_test, model):
    predictions = model.predict(X_test)

    # Compute metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    mape = compute_mape(y_test, predictions)

    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")

    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"R-squared : {r2:.2f}")
    if r2 >= 90:
        bar_color = 'green'
    elif r2 >= 70:
        bar_color = 'yellow'
    else:
        bar_color = 'red'

    progress_html = f"""
    <div style="position: relative; width: 100%; height: 25px; background-color: #f0f0f0; border-radius: 5px;">
        <div style="position: absolute; width: {r2}%; height: 100%; background-color: {bar_color}; border-radius: 5px; transition: width 0.5s;"></div>
        <div style="position: absolute; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; font-weight: bold; color: black ;">
            {r2:.2f}%
        </div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
