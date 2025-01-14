from sklearn.metrics import (
    fbeta_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Initialize the Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
    }
    # Set up hyperparameter tuning with GridSearchCV (optional)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='f1_weighted',  # Handles multiclass by weighting each class
        n_jobs=-1
    )
    # Fit the model on training data
    grid_search.fit(X_train, y_train)
    logger.info(f"Best Model Parameters: {grid_search.best_params_}")
    # Return best model from grid search
    return grid_search.best_estimator_


def compute_model_metrics(y, preds):
    """
    Validates the machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def compute_performance_on_slices(df, feature, y_test, preds):
    """
    Compute performance metrics for slices of data based
    on a categorical feature.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the test data with features as columns.
    feature : str
        Categorical feature to evaluate slices on.
    y_test : np.array
        True binary labels for the data.
    preds : np.array
        Predicted binary labels for the data.

    Returns
    -------
    pandas.DataFrame
        DataFrame with metrics for each slice:
            - `slice_value`: unique values of the feature.
            - `n_samples`: number of samples in the slice.
            - `precision`: precision score for the slice.
            - `recall`: recall score for the slice.
            - `fbeta`: F1 score for the slice.
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' is not in the DataFrame.")

    # Group data by feature values and compute metrics for each slice
    results = [
        {
            "slice_value": value,
            "n_samples": len(y_test[(mask := (df[feature] == value))]),
            "precision": precision_score(y_test[mask], preds[mask], zero_division=0),
            "recall": recall_score(y_test[mask], preds[mask], zero_division=0),
            "fbeta": f1_score(y_test[mask], preds[mask], zero_division=0),
        }
        for value in df[feature].unique()
    ]

    return pd.DataFrame(results)
