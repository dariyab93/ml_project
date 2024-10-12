
import os
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

MODEL_FOLDER = "ml_project/run_scripts"

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    # Replace NaN values with 0s
    data = data.fillna(0)

    # Remove rows where pr_before or pr_after exceeds 1
    data = data[(data['pr_before'] <= 1) & (data['pr_after'] <= 1)]

    # Remove extra single quotes from genotype and treatment columns
    data['genotype'] = data['genotype'].str.replace("'", "", regex=True).str.strip()
    data['treatment'] = data['treatment'].str.replace("'", "", regex=True).str.strip()
    data['nmj_id'] = data['nmj_id'].str.replace("'", "", regex=True).str.strip()

    # Convert genotype, treatment, and nmj columns to strings
    data['genotype'] = data['genotype'].astype(str)
    data['treatment'] = data['treatment'].astype(str)
    data['nmj_id'] = data['nmj_id'].astype(str)
    
    return data

def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    # Feature Engineering
    # Change in Pr
    data['delta_pr'] = data['pr_after'] - data['pr_before']

    # Unc13/Brp ratio feature
    data['unc_brp_ratio'] = data['unc_sum'] / data['brp_sum']
    data['unc_brp_ratio'].fillna(0, inplace=True)  # in case of division by 0

    # Interaction term for Unc13 and Brp
    data['unc_brp_interaction'] = data['unc_sum'] * data['brp_sum']  # not using the normalized data initially
    data['unc_brp_interaction_norm'] = data['unc_norm'] * data['brp_norm']

    # Adding new interaction features using sums
    data['pr_brp_interaction'] = data['pr_before'] * data['brp_sum']
    data['pr_unc_interaction'] = data['pr_before'] * data['unc_sum']
    
    return data

def main() -> None:
    # Load and clean the data
    data = pd.read_csv(os.path.join(MODEL_FOLDER, '../data', 'synapses2.csv'))
    data = clean_data(data)
    data = feature_engineering(data)
    
    # Define features and target
    X = data[['pr_before', 'unc_sum', 'brp_sum', 'unc_brp_ratio', 'unc_brp_interaction', 'delta_pr', 'pr_brp_interaction', 'pr_unc_interaction']]
    y = data['genotype']

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Initialize model with stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb_classifier = XGBClassifier(objective='multi:softmax', random_state=42, use_label_encoder=False)

    # Hyperparameter Grid
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # GridSearchCV for hyperparameter tuning
    xgb_grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=xgb_param_grid, cv=skf, scoring='accuracy', verbose=2, n_jobs=-1)

    # Fit the GridSearchCV to the training data
    xgb_grid_search.fit(X_train, y_train)

    # Get the best model
    best_xgb_model = xgb_grid_search.best_estimator_

    # Evaluate the model on the test set
    y_xgb_pred = best_xgb_model.predict(X_test)
    print(f'Best Hyperparameters: {xgb_grid_search.best_params_}')
    print("Accuracy:", accuracy_score(y_test, y_xgb_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_xgb_pred, target_names=label_encoder.classes_))

    # Save the best model
    joblib.dump(best_xgb_model, os.path.join(MODEL_FOLDER, "../model", "synapse_model.pkl"))
    # Save target class names
    joblib.dump(label_encoder.classes_, os.path.join(MODEL_FOLDER, "../model", "target_names.pkl"))

if __name__ == "__main__":
    main()
