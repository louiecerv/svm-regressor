import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

# Define the Streamlit app
def app():
    if "reset_app" not in st.session_state:
        st.session_state.reset_app = False

    st.title('Predicting Housing Cost using the SVM Regressor')
    # Use session state to track the current form
    if "current_form" not in st.session_state:
        st.session_state["current_form"] = 1    

    # Display the appropriate form based on the current form state
    if st.session_state["current_form"] == 1:
        display_form1()
    elif st.session_state["current_form"] == 2:
        display_form2()
    elif st.session_state["current_form"] == 3:
        display_form3()

def display_form1():
    st.session_state["current_form"] = 1
    form1 = st.form("intro")
    form1.subheader('About the Classifier')
    form1.write("""
        (c) 2024 Louie F. Cervantes
        Department of Computer Science
        College of Information and Communications Technology
        West Visayas state University
    """)
                
    form1.write('Replace with the actual description')        
    #insert the rest of the information here

    submit1 = form1.form_submit_button("Start")

    if submit1:
        form1 = [];
        # Go to the next form        
        display_form2()

def display_form2():
    st.session_state["current_form"] = 2
    form2 = st.form("training")
    form2.subheader('Classifier Training')        

    # Load the California housing data
    data = fetch_california_housing()

    # Convert data features to a DataFrame
    feature_names = data.feature_names
    df = pd.DataFrame(data.data, columns=feature_names)
    df['target'] = data.target
    
    form2.write('The housing dataset')
    form2.write(df)


    submit2 = form2.form_submit_button("Train")
    if submit2:        
        # Separate features and target variable
        X = df.drop('target', axis=1)  # Target variable column name
        y = df['target']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features using StandardScaler (recommended)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create and train the SVM regressor
        from sklearn.linear_model import LinearRegression
        lm = LinearRegression()
        lm.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        y_test_pred = lm.predict(X_test_scaled)

        # Evaluate performance using appropriate metrics (e.g., mean squared error, R-squared)
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        form2.text("Mean squared error:", mse)
        form2.text("R-squared:", r2)

        display_form3()

def display_form3():
    st.session_state["current_form"] = 3
    form3 = st.form("prediction")
    form3.subheader('Prediction')
    form3.text('replace with the result of the prediction model.')

    n_clusters = form3.slider(
        label="Number of Clusters:",
        min_value=2,
        max_value=6,
        value=2,  # Initial value
    )

    predictbn = form3.form_submit_button("Predict")
    if predictbn:                    
        form3.text('User selected nclusters = ' + str(n_clusters))

    submit3 = form3.form_submit_button("Reset")
    if submit3:
        st.session_state.reset_app = True
        st.session_state.clear()

if __name__ == "__main__":
    app()
