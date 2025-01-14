"""
Unit tests for FastAPI endpoints
"""

from fastapi.testclient import TestClient
from main import app  # Assuming your FastAPI app is in a file named `main.py`

client = TestClient(app)


# Test GET endpoint
def test_get():
    """Test the GET endpoint."""
    response = client.get("/")
    assert response.status_code == 200, "Expected status code 200."
    assert response.json() == {"message": "Welcome to the Income Prediction API!"}, "Unexpected response content."


# Test POST request for `>50K` inference
def test_post_above_50K():
    """Test POST request with input data predicting '>50K'."""
    input_data = {
        "age": 51,
        "workclass": "private",
        "fnlgt": 293196,
        "education": "Doctorate",
        "education_num": 16,
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 15024,
        "capital_loss": 0,
        "hours_per_week": 60,
        "native_country": "Iran",
    }

    response = client.post("/predict", json=input_data)
    assert response.status_code == 200, "Expected status code 200."
    assert response.json()["salary"] == ">50K", "Expected prediction to be '>50K'."


# Test POST request for `<=50K` inference
def test_post_below_50K():
    """Test POST request with input data predicting '<=50K'."""
    input_data = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 25000,
        "education": "Bachelors",
        "education_num": 12,
        "marital_status": "Never-married",
        "occupation": "Sales",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }

    response = client.post("/predict", json=input_data)
    assert response.status_code == 200, "Expected status code 200."
    assert response.json()["salary"] == "<=50K", "Expected prediction to be '<=50K'."
