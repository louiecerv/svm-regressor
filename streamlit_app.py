import streamlit as st

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
    # insert the rest of the code to train the classifier here        
    form2.write('Display the training result')

    submit2 = form2.form_submit_button("Train")

    if submit2:        
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
