from flask import Flask
import joblib

app = Flask(__name__)

@app.route('/api')
def llamadoapi():
    pipeline = joblib.load('modelo_entrenado.joblib')
    return pipeline