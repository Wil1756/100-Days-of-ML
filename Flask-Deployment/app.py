from flask import Flask
import pandas as pd
import numpy as np
import sklearn 
import joblib
from flask import Flask,render_template,request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])

def predict():
    if request.method=='POST':
        print(request.form.get('alcohol'))
        print(request.form.get('volatile acidity'))
        print(request.form.get('sulphates'))
        print(request.form.get('total sulfur dioxide'))
        
        try:
            alc = float(request.form['alcohol'])
            vol = float(request.form['volatile acidity'])
            sul = float(request.form['sulphates'])
            dio = float(request.form['total sulfur dioxide'])
            
            pred_args = [alc,vol,sul,dio]
            pred_arr = np.array(pred_args)
            preds = pred_arr.reshape(1, -1)
            
            model = open("linear_regression_model.pkl","rb")
            lr_model = joblib.load(model)
            
            model_pred = lr_model.predict(preds)
            model_pred = round(float(model_pred),2)
            
        except ValueError:
            return "Please Enter valid values"
    return render_template('predict.html',prediction=model_pred)
        

if __name__ == '__main__':
    app.run(debug=True)