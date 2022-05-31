from flask import Flask, request, url_for, redirect, render_template
from joblib import load
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
model = load('/opt/airflow/model/animalModel.pkl')

cols = ['hair', 'feather', 'egg', 'milk', 'airborne', 'aquatic', 
'predator', 'toothed', 'backbone', 'breathes',
'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    predict_data = np.array(features)
    data = pd.DataFrame([predict_data], columns = cols)
    prediction = model.predict(data.values)
    return render_template('home.html', pred='The Animal Type is {}'.format(prediction))

@app.route('/prediction')
def prediction():
    list = request.args.getlist('list', type=int)
    predicts = model.predict([list])
    return "The Animal Type is " + str(predicts)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
