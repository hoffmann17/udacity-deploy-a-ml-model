"""
Unit tests for ml.model.py
Author: Alexander Hoffmann
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data

@pytest.fixture
def data_path():
    return "./data/census.csv"

@pytest.fixture
def data(data_path):
    """Load the Training Data and return dataframe"""
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    return df

@pytest.fixture
def cat_features():
    """Return the categorical features"""
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return cat_features

@pytest.fixture
def perform_train_test_split(data):
    """Load the Training Data and return dataframe"""
    df = data 
    train, test = train_test_split(df, test_size=0.20)
    return train, test

@pytest.fixture
def process_training_data(perform_train_test_split, cat_features):
    """Process Training Data and return it"""

    train, test = perform_train_test_split
    
    X_train, y_train, encoder, lb = process_data(
                                        train, 
                                        categorical_features=cat_features, 
                                        label="salary", 
                                        training=True
                                    )
    return X_train, y_train, encoder, lb

@pytest.fixture
def process_test_data(perform_train_test_split, cat_features, process_training_data):
    """Process Test Data and return it """

    train, test = perform_train_test_split
    X_train, y_train, encoder, lb = process_training_data
    
    X_test, y_test, encoder, lb = process_data(
                                        test, 
                                        categorical_features=cat_features, 
                                        label="salary", 
                                        training=False, 
                                        encoder=encoder, 
                                        lb=lb
                                    )
    return X_test, y_test

@pytest.fixture
def trained_model(process_training_data):
    """Trains and returns a RandomForest model for testing."""
    X_train, y_train, encoder, lb = process_training_data
    model = train_model(X_train, y_train)
    return model

#### Tests ####

def test_categorical_features_exist(data, cat_features):

    df = data
    features = cat_features
    """Test that all required categorical features are present in the dataframe."""
    missing_features = [feature for feature in features if feature not in df.columns]
    assert not missing_features, f"Missing features: {missing_features}"


def test_train_model(process_training_data):
    """Tests the `train_model` function."""
    
    X_train, y_train, encoder, lb = process_training_data

    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Returned model is not a RandomForestClassifier."
    assert hasattr(model, "predict"), "Trained model lacks the `predict` method."


def test_compute_model_metrics():
    """Tests the `compute_model_metrics` function."""
    y_true = np.random.choice([0, 1], size=100)
    y_pred = np.random.choice([0, 1], size=100)
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert 0 <= precision <= 1, "Precision is out of bounds."
    assert 0 <= recall <= 1, "Recall is out of bounds."
    assert 0 <= fbeta <= 1, "F1 score is out of bounds."


def test_inference(trained_model, process_test_data):
    """Tests the `inference` function."""
    model = trained_model
    X_test, y_test = process_test_data
    preds = inference(model, X_test)
    assert len(preds) == len(X_test), "Inference output does not match input size."
    assert all(isinstance(p, (np.integer, int)) for p in preds), "Inference output contains non-integer values."
