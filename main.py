from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model and label encoder
with open('./iris_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('./label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()

    # Extract features from the JSON data
    sepal_length = data['SepalLengthCm']
    sepal_width = data['SepalWidthCm']
    petal_length = data['PetalLengthCm']
    petal_width = data['PetalWidthCm']
    
    # Format input for the model
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make a prediction
    predicted_class = model.predict(input_features)
    predicted_species = label_encoder.inverse_transform(predicted_class)

    # Return the result as JSON
    return jsonify({'Predicted Species': predicted_species[0]})

@app.route('/test', methods=['GET'])
def test():

    # Return the result as JSON
    return jsonify({'test result': '6 ziit'})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
