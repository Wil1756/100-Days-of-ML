from flask import Flask, request
import numpy as np
import pandas as pd
import pickle
from flasgger import Swagger
import joblib

app = Flask(__name__)
Swagger(app)

@app.route('/predict', methods=['GET','POST'])
def predict():
    """
    Returns the predicted value from the ML model
    ---
    parameters:
      - name: alcohol
        in: query
        type: number
        required: true
      - name: volatile acidity
        in: query
        type: number
        required: true
      - name: sulphates
        in: query
        type: number
        required: true
      - name: total sulfur dioxide
        in: query
        type: number
        required: true
    responses:
        200:
            description: Prediction successful
        500:
            description: Prediction failed
    """
    if request.method == 'POST':
        try:
            alc = float(request.args['alcohol'])
            vol = float(request.args['volatile acidity'])
            sul = float(request.args['sulphates'])
            dio = float(request.args['total sulfur dioxide'])
            
            pred_args = [alc, vol, sul, dio]
            pred_arr = np.array(pred_args)
            preds = pred_arr.reshape(1, -1)
            
            model = open("linear_regression_model.pkl", "rb")
            lr_model = joblib.load(model)
            
            model_pred = lr_model.predict(preds)
            model_pred = round(float(model_pred), 2)
            
        except Exception as e:
            return str(e), 500
        
        return str("The predicted value is: " + str(model_pred)), 200
    
    return "Please use POST method for predictions", 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')