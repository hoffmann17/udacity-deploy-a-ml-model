from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

from ml.logger import logger
from ml.data import process_data

# Load the trained model and encoders
model_path = "model/trained_model.joblib"
encoder_path = "model/encoder.joblib"
label_binarizer_path = "model/label_binarizer.joblib"

model = joblib.load(model_path)
encoder = joblib.load(encoder_path)
lb = joblib.load(label_binarizer_path)

# FastAPI instance
app = FastAPI()


# Define the Pydantic model for input data
class InferenceInput(BaseModel):
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str
    age: int
    education_num: int
    fnlgt: int
    capital_gain: int
    capital_loss: int
    hours_per_week: int

    class Config:
        schema_extra = {
            "example": {
                "workclass": "Private",
                "education": "Bachelors",
                "marital_status": "Never-married",
                "occupation": "Prof-specialty",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "native_country": "United-States",
                "age": 35,
                "fnlgt": 77516,
                "education_num": 13,
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 40,
            }
        }


# Define the POST request output schema
class InferenceOutput(BaseModel):
    income: str


@app.get("/")
def root():
    return {"message": "Welcome to the Income Prediction API!"}


@app.post("/predict")
def predict(data: InferenceInput):
    try:
        input_data = {
            'age': data.age,
            'workclass': data.workclass,
            'fnlgt': data.fnlgt,
            'education': data.education,
            'education-num': data.education_num,
            'marital-status': data.marital_status,
            'occupation': data.occupation,
            'relationship': data.relationship,
            'race': data.race,
            'sex': data.sex,
            'capital-gain': data.capital_gain,
            'capital-loss': data.capital_loss,
            'hours-per-week': data.hours_per_week,
            'native-country': data.native_country,
        }

        df = pd.DataFrame(input_data, index=[0])

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

        # Process the Input Dataframe with the process_data function
        X_sample, _, _, _ = process_data(
            df,
            categorical_features=cat_features,
            training=False,
            encoder=encoder,
            lb=lb
        )

        # Make predictions
        prediction = model.predict(X_sample)

        logger.info(f"Prediction: {prediction[0]}, Input Data: {input_data}")

        # Convert prediction to label and add to data output
        if prediction[0] > 0.5:
            salary = '>50K'
        else:
            salary = '<=50K'

        input_data['salary'] = salary
        return input_data

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
