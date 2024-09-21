from flask import Flask
import pandas as pd
import numpy as np
import sklearn 
import joblib


app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, world!</p>"

if __name__ == '__main__':
    app.run(debug=True)