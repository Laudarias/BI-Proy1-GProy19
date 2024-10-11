from flask import Flask, request, jsonify, render_template_string
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

@app.route('/')
def home():
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>API de Predicción</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    text-align: center;
                    margin: 0;
                    padding: 20px;
                }
                h1 {
                    color: #333;
                }
                form {
                    display: inline-block;
                    margin-top: 20px;
                    padding: 20px;
                    background-color: #fff;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    border-radius: 10px;
                }
                input[type="file"] {
                    padding: 10px;
                    margin: 10px 0;
                    display: block;
                }
                input[type="submit"] {
                    background-color: #007BFF;
                    color: #fff;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                }
                input[type="submit"]:hover {
                    background-color: #0056b3;
                }
                p {
                    color: red;
                }
            </style>
        </head>
        <body>
            <h1>Bienvenido a la API de Predicción</h1>
            <p>Sube un archivo JSON para obtener predicciones basadas en los textos proporcionados.</p>
            {% if message %}
                <p>{{ message }}</p>
            {% endif %}
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".json" required><br>
                <input type="submit" value="Predecir">
            </form>
            <h2>Reentrenar el Modelo</h2>
            <form action="/retrain" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".json" required><br>
                <input type="submit" value="Reentrenar">
            </form>
        </body>
        </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    if not pipeline:
        logging.error("El modelo no está disponible.")
        return jsonify({'error': 'El modelo no está disponible en este momento.'}), 500
    
    file = request.files['file']

    try:
        json_data = json.load(file)
    except Exception as e:
        logging.error(f"Error al leer el archivo JSON: {str(e)}")
        return jsonify({'error': 'El archivo no se pudo leer. Asegúrate de que sea un archivo .json válido.'}), 400
    
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
            'comentario': texto,
            'predicción': int(predictions[i]), 
            'probabilidad': float(max(probabilities[i]))  
        })

    return jsonify(results)

@app.route('/results', methods=['POST'])
def results():

    return jsonify({'message': 'Esta función aún no está implementada.'})
@app.route('/retrain', methods=['POST'])
def retrain():
    global pipeline  

    if not pipeline:
        logging.error("El modelo no está disponible.")
        return jsonify({'error': 'El modelo no está disponible en este momento.'}), 500
    
    file = request.files['file']

    try:
        json_data = json.load(file)
    except Exception as e:
        logging.error(f"Error al leer el archivo JSON: {str(e)}")
        return jsonify({'error': 'El archivo no se pudo leer. Asegúrate de que sea un archivo .json válido.'}), 400
    
    try:
        df_nuevos_datos = pd.DataFrame(json_data)
    except Exception as e:
        logging.error(f"Error al convertir los datos: {str(e)}")
        return jsonify({'error': 'Los datos no son válidos. Asegúrate de que estén en el formato correcto.'}), 400
    
    required_columns = ['Textos_espanol', 'sdg']
    if not all(col in df_nuevos_datos.columns for col in required_columns):
        return jsonify({'error': "El DataFrame debe contener las columnas 'Textos_espanol' y 'sdg'."}), 400
    
    try:
        df_datos_existentes = pd.read_excel('datos.xlsx') 
    except Exception as e:
        logging.error(f"Error al leer el archivo datos.xlsx: {str(e)}")
        return jsonify({'error': 'No se pudo leer el archivo datos.xlsx.'}), 500
    
    if not all(col in df_datos_existentes.columns for col in required_columns):
        return jsonify({'error': "El archivo 'datos.xlsx' debe contener las columnas 'Textos_espanol' y 'sdg'."}), 400
    
    try:

        df_combinados = pd.concat([df_datos_existentes, df_nuevos_datos], ignore_index=True)

        df_combinados.to_excel('datos.xlsx', index=False)  
    except Exception as e:
        logging.error(f"Error al combinar y guardar los datos: {str(e)}")
        return jsonify({'error': 'No se pudieron combinar y guardar los datos.'}), 500
    
    X_combinados = df_combinados['Textos_espanol']
    y_combinados = df_combinados['sdg']

    pipeline.fit(X_combinados, y_combinados)

    joblib.dump(pipeline, 'modelo_actualizado.joblib')

    y_pred = pipeline.predict(X_combinados)

    precision = precision_score(y_combinados, y_pred, average='weighted')
    recall = recall_score(y_combinados, y_pred, average='weighted')
    f1 = f1_score(y_combinados, y_pred, average='weighted')
    
    metrics = {
        'precision': float(precision), 
        'recall': float(recall),        
        'f1_score': float(f1)           
    }

    return jsonify(metrics)

if __name__ == '__main__':
    app.run(debug=True)
