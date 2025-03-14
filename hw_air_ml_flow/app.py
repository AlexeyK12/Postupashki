from flask import Flask, request, jsonify

import mlflow
import numpy as np

app = Flask(__name__)

mlflow.set_tracking_uri('http://localhost:5000')
model_uri = 'models:/IrisApp/Production'
model = mlflow.pyfunc.load_model(model_uri)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data['sepal_length'],
                         data['sepal_width'],
                         data['petal_length'],
                         data['petal_width']]).reshape(1, -1)

    prediction = model.predict(features).tolist()
    return jsonify({'class': prediction[0]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# curl -X POST http:http://127.0.0.1/predict -H "Content-Type:application/json" -d '{}'