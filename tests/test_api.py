import pytest
from fastapi.testclient import TestClient
from main import app  # Assuming your FastAPI app is in a file named `app.py`

client = TestClient(app)

# Test GET endpoint
def test_get():
    response = client.get("/")
    
    # Assert status code is 200
    assert response.status_code == 200
    
    # Assert response content
    assert response.json() == {"message": "Welcome to the Income Prediction API!"}

# Test POST request for `>50K` inference
def test_post_above_50K():
    # Define input data that would likely result in '>50K' prediction
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
        "native_country": "Iran"
    }

    response = client.post("/predict", json=input_data)
    
    # Assert status code is 200
    assert response.status_code == 200
    
    # Assert that the prediction is '>50K'
    assert response.json()['salary'] == '>50K'

# Test POST request for `<=50K` inference
def test_post_below_50K():
    # Define input data that would likely result in '<=50K' prediction
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
        "native_country": "United-States"
    }

    response = client.post("/predict", json=input_data)
    
    # Assert status code is 200
    assert response.status_code == 200
    
    # Assert that the prediction is '<=50K'
    assert response.json()['salary'] == '<=50K'
