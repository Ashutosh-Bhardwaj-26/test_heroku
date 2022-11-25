from flask import Flask, request , jsonify
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data 
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
                             
@app.route('/')
def index():

    #ticker = request.args['ticker']
    df = yf.Ticker('AAPL').history(period='max', # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
                                   interval='1d', # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
                                   actions=False)
    #ticker = request.args['ticker']
    data_dict = dict()
    for col in df.columns:
        data_dict[col] = df[col].values.tolist()
    plt.plot(df.Close)
    plt.savefig('abc.png')

    with open("abc.png","rb") as img_file:
        my_final = base64.b64encode(img_file.read()).decode('utf8')

    #splitting for test and train

    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
   
    scaler = MinMaxScaler(feature_range=(0,1))
    data_training_array = scaler.fit_transform(data_training)

    #load the model
    model = load_model('Keras_model_lstm.h5')

    past_100_days = data_training.tail(100)
    final_df = past_100_days.append(data_testing, ignore_index = True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])
    
    x_test, y_test = np.array(x_test), np.array(y_test)
    
    y_predicted = model.predict(x_test)
    scaler = scaler.scale_

    scale_factor = 1/scaler[0]
    y_predicted = y_predicted * scale_factor 
    y_test = y_test * scale_factor

    #final
    plt.figure(figsize=(12,6))
    plt.plot(y_test, 'b', label = 'Original Price')
    plt.plot(y_predicted, 'r', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    plt.savefig('abcd.png')

    with open("abcd.png","rb") as img_file:
        my_final_predicted = base64.b64encode(img_file.read()).decode('utf8')



    return "HEy"



if __name__ == "__main__":
    app.run(debug=True)  