from flask import Flask, request , jsonify
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data 
import yfinance as yf
from keras.models import load_model
from datetime import date
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
                             
@app.route('/')
def index():

    #ticker = request.args['ticker']




    return "HEy"



if __name__ == "__main__":
    app.run(debug=True)  