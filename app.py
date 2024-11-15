from flask import Flask, render_template, request,jsonify
import numpy as np
import yfinance as yf
import pickle

app= Flask(__name__)
# Load the pickle model
with open('trained_model.pkl','rb') as f:
    model = pickle.load(f)
with open('scaler.pkl','rb') as f:
    scaler=pickle.load(f)

def preprocess_input(ticker,start_date,end_date):
    df = yf.download(ticker,start=start_date,end=end_date)
    close_prices = df['Close'].values
    scaled_input_data = scaler.transform(close_prices.reshape(-1,1))
    return scaled_input_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    ticker = request.form['ticker']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    #preprocess input data
    input_data = preprocess_input(ticker,start_date,end_date)
    input_data = np.reshape(input_data,(-1,1))

    predictions_scaled = model.predict(input_data)

    predictions_original = scaler.inverse_transform(predictions_scaled)

    return jsonify({'ticker': ticker,'predictions': predictions_original.tolist()})
   

if __name__ == '__main__':
    app.run(debug=True)