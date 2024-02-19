# Script to train machine learning model.

from sklearn.model_selection import train_test_split

import pandas as pd
import joblib
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..' , '..'))
sys.path.insert(0, root_dir)

from starter.starter.ml.data import process_data
from starter.starter.ml.model import train_model,compute_model_metrics , inference

# Add the necessary imports for the starter code.

# Get the current directory
current_directory = os.path.dirname(__file__)

# Define the path to Model.pkl
data_path = os.path.join(current_directory, '..', 'data', 'census.csv')

data = pd.read_csv(data_path)
data.rename(columns=lambda x: x.replace(' ', ''), inplace=True)
# Assuming test_data is your test dataset

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.10)

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
model_path = os.path.join(current_directory, '..', 'model', 'trained_model.pkl')
encoder_path = os.path.join(current_directory, '..', 'model', 'encoder.pkl')
lb_path = os.path.join(current_directory, '..', 'model', 'lb.pkl')

# Save the trained model to disk
joblib.dump(model, model_path)
joblib.dump(encoder, encoder_path)
joblib.dump(lb, lb_path)

# Proces the test data with the process_data function.
# model = joblib.load("nd0821-c3-starter-code//starter//model//trained_model.pkl")
# print("model loaded")

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False , encoder = encoder ,lb = lb
)
print("processed data")
print(test.head())
print(test.info())
predictions = inference(model , X_test)
print("inference done")
print(predictions)

print(compute_model_metrics(y_test,predictions))

