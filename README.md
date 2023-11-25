# Titanic Pipeline
==============================

## Training
* Podemos configurar el training en src-config-config.py
* Podemos iniciar el training con `python -m train.train_model`
* Posteriormente, dirigirse a la carpeta `/models`, identificar el modelo recién entrenado

## Serving/Inference (Falta por implementar)
* Podemos iniciar el servidor para las predicciones con `uvicorn server.app:app --reload` .
* Podemos rquerir una prediccion mandando un `POST` (se puede usar Postman) a la url http://127.0.0.1:8000/prediction con un JSON en el Body en el siguiente formato:
```
{
    "pclass": 1,
    "name": "Allen, Miss. Elisabeth Walton",
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
```
## Lista de Logs
| Path          | Description   | Severity   |
| ------------- | ------------- | ------------- |
| `titanic/server/app.py`  | Log para registrar para el modelo: id, version, datos de entrada, datos procesados y prediccion. Para el sistema: timepo de ejecución y memoria utilizada. | INFO |
| `titanic/server/app.py`  | Carga del modelo | CRITICAL |
* 

## Opcional
* Falta agregar una página inicial con la documentación
