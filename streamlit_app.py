import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
@st.cache_data
def load_data():
    sonar_data = pd.read_csv('sonardata.csv', header=None)
    return sonar_data

def train_model():
    sonar_data = load_data()

    # Separate data and labels
    X = sonar_data.drop(columns=60, axis=1)
    Y = sonar_data[60]

    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

    # Train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    return model, X_test, Y_test

# Initialize the model
model, X_test, Y_test = train_model()

# Streamlit app setup
st.title('Rock vs Mine Prediction')
st.write("This app predicts whether the object is a rock or a mine based on sonar data.")

# Input data fields
st.header('Input Features')
user_input = st.text_area("Enter all 60 features separated by commas (e.g., 0.02,0.03,...,0.04):", "")

if user_input:
    try:
        # Parse the input into a numpy array
        input_data_as_numpy_array = np.array([float(x.strip()) for x in user_input.split(',')])

        if len(input_data_as_numpy_array) != 60:
            st.error("Please ensure you enter exactly 60 features.")
        else:
            # Reshape the data for prediction
            input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

            # Make prediction on the input data
            prediction = model.predict(input_data_reshape)
            if prediction[0] == 'R':
                st.success('The object is a Rock.')
            else:
                st.success('The object is a Mine.')
    except ValueError:
        st.error("Please enter valid numeric values separated by commas.")

# Display test accuracy
if st.checkbox('Show Test Accuracy'):
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    st.write(f'Accuracy on test data: {test_data_accuracy * 100:.2f}%')
