import os
import joblib
import pandas as pd

MODEL_FOLDER = "ml_project/run_scripts"

class SynapseClassPredictor:
    def __init__(self, model_path: str = None, target_names_path: str = None):
        self.model_path = model_path or os.path.join(MODEL_FOLDER, "synapse_model.pkl")
        self.target_names_path = target_names_path or os.path.join(MODEL_FOLDER, "target_names.pkl")
        self._load_model()

    def _load_model(self):
        # Load the model and target names
        self.model = joblib.load(self.model_path)
        self.target_names = joblib.load(self.target_names_path)

    def predict(self, input_data: pd.DataFrame):
        # Ensure that the input data has the correct features
        required_features = [
            'pr_before', 'unc_sum', 'brp_sum', 'unc_brp_ratio',
            'unc_brp_interaction', 'pr_brp_interaction', 'pr_unc_interaction'
        ]
        if not all(feature in input_data.columns for feature in required_features):
            raise ValueError("Missing required features in input data")

        # Make predictions
        predictions = self.model.predict(input_data)
        prediction_labels = [self.target_names[pred] for pred in predictions]
        return prediction_labels

if __name__ == "__main__":
    # Example usage
    data = pd.DataFrame({
        'pr_before': [0.1],
        'unc_sum': [0.5],
        'brp_sum': [0.4],
        'unc_brp_ratio': [1.25],
        'unc_brp_interaction': [0.2],
        'pr_brp_interaction': [0.04],
        'pr_unc_interaction': [0.05]
    })

    predictor = SynapseClassPredictor()
    predictions = predictor.predict(data)
    print(f"Predicted class: {predictions[0]}")
