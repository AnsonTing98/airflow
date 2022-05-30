import pandas as pd
from pandas import read_csv
from joblib import dump
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

zoo1_df = read_csv("zoo.csv")
zoo2_df = read_csv("zoo2.csv")
zoo3_df = read_csv("zoo3.csv")
class_df = read_csv("class.csv")

zoo1_2_3_df = pd.concat([zoo1_df, zoo2_df, zoo3_df])
zoo1_2_3_df = zoo1_2_3_df.sort_values(by = ['animal_name'])
zoo_df = zoo1_2_3_df.merge(class_df, how = "left", left_on = "class_type", right_on = "Class_Number")

X = zoo_df.drop(["animal_name", "class_type", "Class_Number", "Number_Of_Animal_Species_In_Class", "Class_Type", "Animal_Names"], axis = 1)
y = zoo_df.Class_Type.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = GaussianNB().fit(X_train.values, y_train)

dump(model, "animalModel.pkl")

