import streamlit as st
import re
import numpy as np
from tensorflow.keras.models import model_from_json

# Load the LSTM model
def load_model():
    # Load the model architecture
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    
    model = model_from_json(loaded_model_json)
    
    # Load the model weights
    model.load_weights("my_model.keras")
    
    # Compile the model with an optimizer and loss function
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Function to predict sentiment
def sentiment_prediction(review, model):
    input_review = [review.lower()]
    input_review = [re.sub(r'[^a-zA-Z0-9\s]', '', x) for x in input_review]
    
    # Process the input into the required format for the model
    processed_input = np.array(input_review)  

    # Get the prediction from the model
    sentiment = model.predict(processed_input)
    
    # Interpret the result
    if sentiment > 0.5:
        return "Positive"
    else:
        return "Negative"

# Function to run the Streamlit app
def run():
    st.title("Sentiment Analysis - LSTM Model")
    
    # HTML template ( empty)
    html_temp = """
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # User input
    review = st.text_input("Enter the Review")
    
    # Prediction result
    prediction = ""

    # Button to trigger the sentiment prediction
    if st.button("Predict Sentiment"):
        model = load_model()  
        prediction = sentiment_prediction(review, model) 
        st.success("The sentiment predicted by the model: {prediction}")

# Entry point for running the app
if __name__ == '__main__':
    run()
