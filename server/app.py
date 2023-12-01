from fastapi import FastAPI
import logging
import joblib
import os
import pydantic
import pandas as pd

# Logging setup
logging.basicConfig(filename=os.path.abspath('./logs/api.log'), level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ML model setup
def load_model():
    model = joblib.load(os.path.abspath('./models/model.sav'))
    logging.critical("Model loaded")
    return model

class PredInput(pydantic.BaseModel):
    pclass:int
    name:str
    sex:str
    age:str
    sibsp: int
    parch: int
    ticket: str
    fare: str
    cabin: str
    embarked: str
    boat: str
    body: str
    home_dest: str
class PredOutput(pydantic.BaseModel):
    prediction_class:int
    type_of_model:str

def input_formater(input:PredInput):
    X = input.__dict__
    X['index'] = 0
    return pd.DataFrame(X,index=[0])

# API setup
app = FastAPI()

@app.get('/')
async def root():
    return {'message': 'Bienvenido al predictor de sobrevivientes del Titanic'}

@app.post('/predict')
async def predict_survivor(input:PredInput) -> PredOutput:
    model = load_model()
    X = input_formater(input).drop(columns=['index','home_dest'])
    pred = model.predict(X)
    logging.info(f"Input de la predicción es {X.values}")
    logging.info(f"La prediccion es {pred[0]}")
    return {'prediction_class': pred,
            'type_of_model': 'LogisticRegressor'}