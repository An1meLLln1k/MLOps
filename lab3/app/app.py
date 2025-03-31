from flask import Flask, request, jsonify
from model import Model

app = Flask(__name__)
model = Model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'values' not in data:
        return jsonify({'error': 'Missing "values" in request'}), 400

    try:
        values = data['values']
        predictions = model.predict(values)
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
