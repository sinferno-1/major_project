from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
# import csv
from datetime import datetime

app = Flask(__name__)
model = load_model('lstm.keras')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['input']
    
    # Convert input data to numpy array
    input_data = np.array(data)
    
    # Ensure the shape of input data is correct
    if input_data.shape != (24, 1, 16):
        return jsonify({'error': 'Invalid input shape'}), 400

    # Predict the next timestep
    next_timestep_prediction = model.predict(input_data)

    # Convert the prediction to a list for JSON serialization
    prediction_list = next_timestep_prediction.tolist()
    next_step_prediction = next_timestep_prediction[0]
    
    # Save prediction with timestamp to CSV file
    # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # prediction_with_timestamp = [timestamp] + next_step_prediction.tolist()
    
    return jsonify({'prediction': next_step_prediction.tolist()}), 200


if __name__ == '__main__':
    # app.run("0.0.0.0", port=80)
    app.run()
