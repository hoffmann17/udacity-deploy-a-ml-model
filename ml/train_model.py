# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import logging

from model import train_model, compute_model_metrics, inference, compute_performance_on_slices
from data import process_data


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Initialize Variables
data_path = "data/census.csv"
model_path = "model"
slice_path = "./slice_output.csv"

# Add code to load in the data.
data = pd.read_csv(data_path)
data.columns = data.columns.str.strip()  # Clean column names otherwise pandas would add an empty space infront of the column name causing errors

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

# Process the training data with the process_data function
logger.info("Process Training data")
X_train, y_train, encoder, lb = process_data(
    train, 
    categorical_features=cat_features, 
    label="salary", 
    training=True
)

# Process the test data with the process_data function.
logger.info("Process Test data")
X_test, y_test, encoder, lb = process_data(
    test, 
    categorical_features=cat_features, 
    label="salary", 
    training=False, 
    encoder=encoder, 
    lb=lb
)

# Train and save a model.

## Train the model
model = train_model(X_train, y_train)

## Test the model
preds = inference(model, X_test)

logger.info(preds)

## Get Model Performance
precision, recall, fbeta = compute_model_metrics(y_test, preds)

logger.info(f"Model Performance: Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {fbeta:.4f}")

# Save the trained model and encoder/label binarizer for later use.
joblib.dump(model, f"{model_path}/trained_model.joblib")
joblib.dump(encoder, f"{model_path}/encoder.joblib")
joblib.dump(lb, f"{model_path}/label_binarizer.joblib")

for feature in cat_features:
    df_performance = compute_performance_on_slices(test, feature, y_test, preds)
    df_performance.to_csv(slice_path,  mode='a', index=False)
    logging.info(f"Performance on slice {feature}")
    logging.info(df_performance)