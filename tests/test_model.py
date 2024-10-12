import json
import pandas as pd
from synapse_class_predictor import SynapseClassPredictor

# Load test examples from JSON file
def load_test_examples(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

# Test the model with the test examples
if __name__ == "__main__":
    # Load test examples
    test_examples_path = "ml_project/run_scripts/test_examples.json"
    test_data = load_test_examples(test_examples_path)

    # Initialize the predictor
    predictor = SynapseClassPredictor()

    # Make predictions
    predictions = predictor.predict(test_data)

    # Print predictions
    for i, prediction in enumerate(predictions):
        print(f"Test Example {i + 1}: Predicted class - {prediction}")
