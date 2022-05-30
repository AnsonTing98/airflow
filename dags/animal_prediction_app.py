import pandas as pd
import numpy as np
from joblib import dump
from joblib import load
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from flask import Flask, request, url_for, redirect, render_template

from datetime import timedelta
from airflow import DAG
from airflow.plugins_manager import AirflowPlugin
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import datetime
import os

dag_path = os.getcwd()

def merge_data():
   zoo1_df = pd.read_csv("raw_data/zoo.csv")
   zoo2_df = pd.read_csv("raw_data/zoo2.csv")
   zoo3_df = pd.read_csv("raw_data/zoo3.csv")
   class_df = pd.read_csv("raw_data/class.csv")

   zoo1_2_3_df = pd.concat([zoo1_df, zoo2_df, zoo3_df])
   zoo1_2_3_df = zoo1_2_3_df.sort_values(by = ['animal_name'])
   zoo_df = zoo1_2_3_df.merge(class_df, how = "left", left_on = "class_type", right_on = "Class_Number")

   zoo_df.to_csv("processed_data/zoo-merged.csv", index = False)

def training_model():
   df = pd.read_csv("processed_data/zoo-merged.csv")
   
   X = df.drop(["animal_name", "class_type", "Class_Number", "Number_Of_Animal_Species_In_Class", "Class_Type", "Animal_Names"], axis = 1)
   y = df.Class_Type.values
 
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

   model = GaussianNB().fit(X_train.values, y_train)

   dump(model, "model/animalModel.pkl")

def test_accuracy():
   model = load("model/animalModel.pkl")

   df = pd.read_csv("processed_data/zoo-merged.csv")

   X = df.drop(["animal_name", "class_type", "Class_Number", "Number_Of_Animal_Species_In_Class", "Class_Type", "Animal_Names"], axis = 1)
   y = df.Class_Type.values

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

   y_pred = model.predict(X_test.values)

   print("Train Accuracy : ", model.score(X_train.values, y_train))
   print("Test Accuracy : ", model.score(X_test.values, y_test))
   print("Confusion Matrix : \n", confusion_matrix(y_test, y_pred, labels=np.unique(y_pred)))
   print("classification report : \n", classification_report(y_test, y_pred, labels=np.unique(y_pred)))

def prediction():
   model = load("model/animalModel.pkl")
   
   def ml_predictions(x):
      result = model.predict(x)
      return result

   arr = [[0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 2, 1, 1, 0]]
   arr2 = [[1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 4, 0, 0, 1]]

   print(arr)
   print("First prediction: ", ml_predictions(arr))
   print(arr2)
   print("Second prediction: ", ml_predictions(arr2))

def create_app():
   app = Flask(__name__)
   model = load('model/animalModel.pkl')

   cols = ['hair', 'feather', 'egg', 'milk', 'airborne', 'aquatic',
           'predator', 'toothed', 'backbone', 'breathes',
           'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize']

   @app.route('/')
   def home():
      return render_template("templates/home.html")

   @app.route('/predict', methods=['POST'])
   def predict():
      features = [x for x in request.form.values()]
      predict_data = np.array(features)
      data = pd.DataFrame([predict_data], columns = cols)
      prediction = model.predict(data.values)
      return render_template('templates/home.html', pred='The Animal Type is {}'.format(prediction))

   @app.route('/prediction')
   def prediction():
      list = request.args.getlist('list', type=int)
      predicts = model.predict([list])
      return "The Animal Type is " + str(predicts)

   if __name__ == '__main__':
      return app

default_args={
   'owner': 'airflow',
   'start_date': datetime(2022, 5, 1)
}

prediction_dag = DAG(
   'animal_prediction_app',
      default_args = default_args,
      description = "Animal Class Predict Application",
      schedule_interval = "@daily", 
      catchup=False
)

task_merge_data = PythonOperator(
   task_id = 'merge_data',
   provide_context = True,
   python_callable = merge_data,
   dag = prediction_dag
)

task_training_model = PythonOperator(
   task_id = 'training_model',
   provide_context = True,
   python_callable = training_model,
   dag = prediction_dag
)

task_test_accuracy = PythonOperator(
   task_id = 'test_accuracy',
   provide_context = True,
   python_callable = test_accuracy,
   dag = prediction_dag
)

task_prediction = PythonOperator(
   task_id = 'prediction',
   provide_context = True,
   python_callable = prediction,
   dag = prediction_dag
)

task_create_application = PythonOperator(
   task_id = 'create_application',
   provide_context = True,
   python_callable = create_app,
   dag = prediction_dag
)


task_run_application = BashOperator(
   task_id = 'run_application',
   depends_on_past=False,
   bash_command = "python /opt/airflow/script/app.py",
   dag = prediction_dag
)

task_merge_data >> task_training_model >> [task_test_accuracy,task_prediction] >> task_create_application >> task_run_application
