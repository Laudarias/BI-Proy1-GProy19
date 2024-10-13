from flask import Flask, request, jsonify, render_template_string, send_file
import joblib
import pandas as pd
import json
from sklearn.metrics import f1_score, precision_score, recall_score
from modelo import TextPreprocessing
import logging
import numpy as np  
import requests

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
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                    background-color: #fff;
                    border-radius: 10px;
                    box-shadow: 0 0 15px rgba(0, 0, 0, 0.2);
                    padding: 20px;
                }
                form {
                    margin-top: 20px;
                    padding: 20px;
                    border-radius: 10px;
                    background-color: #f9f9f9;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }
                h2 {
                    color: #007BFF;
                }
                input[type="file"], textarea {
                    padding: 10px;
                    margin: 10px 0;
                    display: block;
                    width: 100%;
                    border: 1px solid #ccc;
                    border-radius: 5px;
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
                .message {
                    margin-top: 20px;
                    color: #d9534f;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Bienvenido a la API de Clasificación de ODS</h1>
                <p>Sube un archivo XLSX para obtener clasificaciones.</p>
                {% if message %}
                    <p class="message">{{ message }}</p>
                {% endif %}
                
                <form action="/predict" method="post" enctype="multipart/form-data">
                    <h2>Subir archivo XLSX para clasificar</h2>
                    <input type="file" name="file" accept=".xlsx" required>
                    <input type="submit" value="Predecir">
                </form>


                <form action="/retrain" method="post" enctype="multipart/form-data">
                    <h2>Reentrenar el Modelo</h2>
                    <input type="file" name="file" accept=".xlsx" required>
                    <input type="submit" value="Reentrenar">
                </form>
            </div>
        </body>
        </html>
    ''')


@app.route('/predict', methods=['POST'])
def predict():
  
    file = request.files['file']
    df = pd.read_excel(file)

    textos = df.iloc[1:, 0].tolist()
    json_data = [{"Textos_espanol": texto} for texto in textos]
    print(json_data[0])

    url = 'http://localhost:5000/predict'
    response = requests.post(url, json=json_data)

    if response.status_code == 200:
   
        response_data = response.json()
        df_response = pd.DataFrame(response_data)

      
        file_path = 'respuesta_generada.xlsx'
        df_response.to_excel(file_path, index=False)

        return send_file(file_path, as_attachment=True, download_name='respuesta_generada.xlsx', mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    else:
        return jsonify({'error': 'Error al hacer la solicitud al servidor.'}), response.status_code


 
@app.route('/retrain', methods=['POST'])
def retrain():
    global pipeline  

    if not pipeline:
        logging.error("El modelo no está disponible.")
        return jsonify({'error': 'El modelo no está disponible en este momento.'}), 500
    
    file = request.files['file']

    try:

        df_nuevos_datos = pd.read_excel(file)  
    except Exception as e:
        logging.error(f"Error al leer el archivo Excel: {str(e)}")
        return jsonify({'error': 'El archivo no se pudo leer. Asegúrate de que sea un archivo .xlsx válido.'}), 400
    
    required_columns = ['Textos_espanol', 'sdg']
    if not all(col in df_nuevos_datos.columns for col in required_columns):
        return jsonify({'error': "El DataFrame debe contener las columnas 'Textos_espanol' y 'sdg'."}), 400
    

    
    try:
    
        X_nuevos = df_nuevos_datos['Textos_espanol']
        y_nuevos = df_nuevos_datos['sdg']

        pipeline.fit(X_nuevos, y_nuevos)

        joblib.dump(pipeline, 'modelo_actualizado.joblib')

        y_pred = pipeline.predict(X_nuevos)

        precision = precision_score(y_nuevos, y_pred, average='weighted')
        recall = recall_score(y_nuevos, y_pred, average='weighted')
        f1 = f1_score(y_nuevos, y_pred, average='weighted')
        
        metrics = {
            'precision': float(precision), 
            'recall': float(recall),        
            'f1_score': float(f1)           
        }

        return jsonify(metrics)
    
    except Exception as e:
        logging.error(f"Error durante el reentrenamiento incremental: {str(e)}")
        return jsonify({'error': 'Error durante el reentrenamiento incremental.'})

if __name__ == '__main__':
    app.run(debug=True,  port=5001)
