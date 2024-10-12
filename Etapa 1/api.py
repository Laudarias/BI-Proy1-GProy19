from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json
from sklearn.metrics import f1_score, precision_score, recall_score
from modelo import TextPreprocessing
import logging
import numpy as np 


logging.basicConfig(level=logging.INFO, filename='app.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')


app = Flask(__name__)

try:
    pipeline = joblib.load('modelo_entrenado.joblib')
except Exception as e:
    logging.error(f"Error al cargar el modelo: {str(e)}")
    pipeline = None


@app.route('/predict', methods=['POST'])
def predict():



    json_data = request.get_json()
    print(json_data)
    if not isinstance(json_data, list) or len(json_data) == 0 or 'Textos_espanol' not in json_data[0]:
        logging.error("El formato del JSON es incorrecto.")
        return jsonify({'error': "El archivo debe contener una lista de objetos con la clave 'Textos_espanol'."}), 400
    
    texts = [item['Textos_espanol'] for item in json_data]

    try:
        predictions = pipeline.predict(texts)
        probabilities = pipeline.predict_proba(texts)
    except Exception as e:
        logging.error(f"Error al realizar la predicción: {str(e)}")
        return jsonify({'error': 'Error en la predicción. Verifica que el modelo y los datos sean correctos.'}), 500
    
    results = []
    for i, texto in enumerate(texts):
        results.append({
            'prediccion': int(predictions[i]), 
            'probabilidad': float(max(probabilities[i]))  
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True,  port=5000)
