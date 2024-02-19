import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, fbeta_score
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..' , '..'))
sys.path.insert(0, root_dir)

from  starter.starter.ml.model import inference , compute_model_metrics , train_model

def test_inference():
    # Define a sample machine learning model (you may need to adjust this)
    class SampleModel:
        def predict(self, X):
            # Dummy predictions for testing
            return np.array([1, 2, 3])

    # Create an instance of the sample model
    model = SampleModel()

    # Define sample input data for prediction
    X = np.array([[1, 2], [3, 4], [5, 6]])

    # Call the inference function with the sample model and input data
    preds = inference(model, X)

    # Define the expected predictions
    expected_preds = np.array([1, 2, 3])

    # Check that the output predictions match the expected predictions
    assert np.array_equal(preds, expected_preds)
    
    
    
# Define a test function
def test_compute_model_metrics():
    # Define sample true labels and predicted labels
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 1, 0, 0, 1])  # Adjusted for testing purposes

    # Call the compute_model_metrics function with the sample labels
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Compute the expected metrics manually
    expected_precision = precision_score(y_true, y_pred, zero_division=1)
    expected_recall = recall_score(y_true, y_pred, zero_division=1)
    expected_fbeta = fbeta_score(y_true, y_pred, beta=1, zero_division=1)

    # Check that the computed metrics match the expected metrics
    assert pytest.approx(precision) == expected_precision
    assert pytest.approx(recall) == expected_recall
    assert pytest.approx(fbeta) == expected_fbeta




# Define a test function
def test_train_model():
    # Generate synthetic data for testing
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Call the train_model function with the sample training data
    model = train_model(X_train, y_train)

    # Check that the returned model is an instance of RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)

    # Check that the model has been trained (i.e., it's not None)
    assert model is not None

    # Optionally, you can add more specific checks on the trained model if needed