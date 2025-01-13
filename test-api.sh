#!/bin/bash

# Define API endpoints
BASE_URL="http://127.0.0.1:8000"
GET_ENDPOINT="$BASE_URL/"
POST_ENDPOINT="$BASE_URL/predict"

# Test GET request
echo "Testing GET endpoint..."
curl -X GET $GET_ENDPOINT -H "Content-Type: application/json"
echo -e "\nGET request completed."

# Test POST request
echo "Testing POST endpoint..."
echo ""
curl -X POST $POST_ENDPOINT \
  -H "Content-Type: application/json" \
  -d '{
        "age": 50,
        "workclass": "Private", 
        "fnlgt": 234721,
        "education": "Doctorate",
        "education_num": 16,
        "marital_status": "Separated",
        "occupation": "Exec-managerial",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
      }'



curl -X POST $POST_ENDPOINT \
  -H "Content-Type: application/json" \
  -d '{
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
      }'

echo ""
echo "\nPOST request completed."

curl -X POST $POST_ENDPOINT \
  -H "Content-Type: application/json" \
  -d '{
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 45,
        "native_country": "United-States"
      }'

echo ""
curl -X POST $POST_ENDPOINT \
  -H "Content-Type: application/json" \
  -d '{
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
      }'

