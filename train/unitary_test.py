import pytest
import joblib
import pandas as pd
from train.train_model import train
import os
import numpy as np

def train_model():
    return [('LogisticRegressor',True),('RandomForest',True)]

@pytest.mark.parametrize('type_of_model, status',train_model())
def test_train_model(type_of_model,status):
    train(model_name=type_of_model)
    try:
        model = joblib.load(os.path.abspath('./models/model.sav'))
        resultado = True
    except FileNotFoundError:
        resultado = False
    assert resultado == status

X1 = {
  "pclass": 1,
  "name": "Allen, Miss. Elizabeth Walton",
  "sex": "female",
  "age": "29",
  "sibsp": 0,
  "parch": 0,
  "ticket": "24160",
  "fare": "211.3375",
  "cabin": "B5",
  "embarked": "S",
  "boat": "2",
  "body": "?",
  "home_dest": "St Louis, MO"
}
X2 = {
  "pclass": 0,
  "name": "Allen, Miss. Elizabeth Walton",
  "sex": "female",
  "age": "36",
  "sibsp": 0,
  "parch": 0,
  "ticket": "242340",
  "fare": "211.3375",
  "cabin": "B5",
  "embarked": "S",
  "boat": "7",
  "body": "?",
  "home_dest": "St Louis, MO"
}
def input_formater(input:dict):
    input['index'] = 0
    return pd.DataFrame(input,index=[0]).drop(columns=['index','home_dest'])

def model_predict():
    return [(input_formater(X1),True),(input_formater(X2),True)]

@pytest.mark.parametrize('X_vars, status',model_predict())
def test_model_predict(X_vars,status):
    model = joblib.load(os.path.abspath('./models/model.sav'))
    try:
        prediction = model.predict(X_vars)
        if type(prediction[0])==np.int64:
            resultado = True
        else:
            resultado = False    
    except FileNotFoundError:
        resultado = False
    assert resultado == status