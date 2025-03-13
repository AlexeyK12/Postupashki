from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# загрузка модели
model = joblib.load('iris_model_catboost.joblib')

# маршрут для корневого пути
@app.route('/')
def home():
    return "Welcome to the Iris Prediction API! Use /predict to make predictions."

# маршрут для предсказаний
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data['sepal_length'],
                         data['sepal_width'],
                         data['petal_length'],
                         data['petal_width']]).reshape(1, -1)

    prediction = model.predict(features).tolist()
    return jsonify({'class': prediction[0]})

# маршрут для favicon.ico
@app.route('/favicon.ico')
def favicon():
    return '', 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)