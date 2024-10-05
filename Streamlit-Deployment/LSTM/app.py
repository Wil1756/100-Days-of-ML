import os
import numpy as np
import pandas as pd
import pickle
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.models
from tensorflow.keras.models import model_from_json
import streamlit
import re
os.environ['TF_CP_MIN_LOG_LEVEL'] ='2'

#UI
def run():
    streamlit.title('Sentiment Analysis - LSTM Model')
    html="""
    """
    streamlit.markdown(html_temp)
    review = streamlit.text_input("Enter the Review")
    prediction=""
    
    if streamlit.button("Predict Sentiment"):
        prediction=sentiment_prediction(review)
    streamlit.success('The sentiment predicted by Model : {}'.format(prediction))
    
    if __name__ =='__main__':
        run()
    