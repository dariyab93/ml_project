import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Load the model and target names
MODEL_FOLDER = "ml_project/model"
model = joblib.load(os.path.join(MODEL_FOLDER, "synapse_model.pkl"))
target_names = joblib.load(os.path.join(MODEL_FOLDER, "target_names.pkl"))

# Create a Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Convert input data to a DataFrame
        input_data = pd.DataFrame(data, index=[0])
        
        # Preprocessing: make sure all required features are present
        required_features = [
            'pr_before', 'unc_sum', 'brp_sum', 'unc_brp_ratio',
            'unc_brp_interaction', 'pr_brp_interaction', 'pr_unc_interaction'
        ]
        if not all(feature in input_data.columns for feature in required_features):
            return jsonify({'error': 'Missing required features in input data'}), 400
        
        # Make predictions
        prediction = model.predict(input_data)
        prediction_label = target_names[prediction[0]]
        
        # Return the prediction
        return jsonify({'prediction': prediction_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
