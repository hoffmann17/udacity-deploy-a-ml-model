import requests
import json

# Define API endpoints
BASE_URL = "https://udacity-ml-deployment-ea5993e3c22e.herokuapp.com"
GET_ENDPOINT = f"{BASE_URL}/"
POST_ENDPOINT = f"{BASE_URL}/predict"

# Test GET request
print("Testing GET endpoint...")
response = requests.get(GET_ENDPOINT, headers={"Content-Type": "application/json"})
print(f"GET response status code: {response.status_code}")
print(f"GET response body: {response.text}\n")

# Test POST request
data = {
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
}

print("Testing POST endpoint...")
response = requests.post(
    POST_ENDPOINT,
    headers={"Content-Type": "application/json"},
    data=json.dumps(data)
)
print(f"POST response status code: {response.status_code}")
print(f"POST response body: {response.text}")
