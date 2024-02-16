# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model,compute_model_metrics , inference
import pandas as pd
import joblib
import os

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv("nd0821-c3-starter-code//starter//data//census.csv")
data.rename(columns=lambda x: x.replace(' ', ''), inplace=True)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
# Train and save a model.
model = train_model(X_train,y_train)
model_filename = 'nd0821-c3-starter-code//starter//model//trained_model.pkl'
encoder_filename = 'nd0821-c3-starter-code//starter//model//encoder.pkl'

# Save the trained model to disk
joblib.dump(model, model_filename)
joblib.dump(encoder, encoder_filename)

# Proces the test data with the process_data function.
# model = joblib.load("nd0821-c3-starter-code//starter//model//trained_model.pkl")
# print("model loaded")

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False , encoder = encoder ,lb = lb
)
print("processed data")

predictions = inference(model , X_test)
print("inference done")
print(predictions)

print(compute_model_metrics(y_test,predictions))

