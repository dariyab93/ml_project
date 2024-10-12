# ML Project: Synapse Classification

## Project Overview
This project aims to develop a predictive model to classify synapses into different genotypes based on their synaptic characteristics. The model was trained using data collected during my Ph.D. research on synaptic behavior in neurons under various conditions. The goal was to predict the genotype (Wild Type, Unc13A-RNAi, Unc13B-RNAi) of synapses using several engineered features.

## Data
The dataset used in this project contains measurements of synaptic strength and protein contents across synapses. Key features include synaptic release probability before drug treatment (`pr_before`), total Unc13 and Brp protein counts (`unc_sum`, `brp_sum`), as well as several interaction features (`unc_brp_ratio`, `unc_brp_interaction`, etc.). The data is stored in the `/data` folder as a CSV file (`synapses2.csv`).

## Repository Structure
The repository has the following structure:
- `data/`: Contains the raw dataset (`synapses2.csv`).
- `run_scripts/`: Contains the scripts used to train and deploy the model.
  - `train_synapse_model.py`: Script to train the classification model.
  - `api.py`: Flask-based API for model deployment.
  - `test_model.py`: Script for testing the model using test examples.
  - `synapse_class_predictor.py`: Utility script for loading the model and making predictions.
- `model/`: Stores the trained model (`synapse_model.pkl`).
- `README.md`: Project overview and instructions.
- `.gitignore`: Specifies which files to ignore in version control.

## Key Features and Engineering
Several features were engineered to improve the model's predictive ability:
- **Delta Pr (`delta_pr`)**: Change in probability of release before and after treatment.
- **Unc13/Brp Ratio (`unc_brp_ratio`)**: The ratio between Unc13 and Brp protein counts.
- **Interaction Terms**: Terms such as `unc_brp_interaction`, `pr_brp_interaction`, and `pr_unc_interaction` were added to capture possible synergistic effects between features.

## Model
The model that performed best in this task was an **XGBoost Classifier**. Hyperparameter tuning using `GridSearchCV` was applied to achieve the best model performance.

## Deployment
The model has been deployed using **Flask**. The `api.py` script creates an API endpoint (`/predict`) to make predictions based on new data.

## Usage
### Training the Model
Run the `train_synapse_model.py` script from the `run_scripts` folder to train the model.
```bash
python run_scripts/train_synapse_model.py
```

### Deploying the Model
To deploy the model, run the `api.py` script:
```bash
python run_scripts/api.py
```

The API will be available at `http://0.0.0.0:5000/predict`. Send a POST request with JSON data to receive predictions.

### Testing the Model
To test the model, run the `test_model.py` script:
```bash
python run_scripts/test_model.py
```

The script will load test examples from `test_examples.json` and provide predictions.

## Evaluation Metrics
The model's performance was evaluated using:
- **Accuracy**: The accuracy of the model across test examples.
- **Classification Report**: Precision, recall, and F1-score for each genotype class.

## Challenges and Learnings
- Predicting **Delta Pr** using regression models was challenging, possibly due to complex biological interactions that aren't fully captured by the available features.
- Classification of synapse genotypes, however, was successful, and the **unc_brp_ratio** feature was highly informative in distinguishing between the different genotypes.

## Future Work
- Further feature engineering may help improve predictions of Delta Pr.
- Exploring other models like **LightGBM** or **neural networks** could further improve classification performance.

## Requirements
To run the code, make sure to install the required Python packages:
```bash
pip install -r requirements.txt
```

## License
This project is licensed under the MIT License.

