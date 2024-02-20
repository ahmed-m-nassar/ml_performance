import sys
import os
import numpy as np
import pytest
from sklearn.metrics import precision_score, recall_score, fbeta_score
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from starter.starter.ml.model import inference
from starter.starter.ml.model import compute_model_metrics
from starter.starter.ml.model import train_model
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)


def test_inference():
    class SampleModel:
        def predict(self, X):
            return np.array([1, 2, 3])

    model = SampleModel()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    preds = inference(model, X)
    expected_preds = np.array([1, 2, 3])
    assert np.array_equal(preds, expected_preds)


def test_compute_model_metrics():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    expected_precision = precision_score(y_true, y_pred, zero_division=1)
    expected_recall = recall_score(y_true, y_pred, zero_division=1)
    expected_fbeta = fbeta_score(y_true, y_pred, beta=1, zero_division=1)
    assert pytest.approx(precision) == expected_precision
    assert pytest.approx(recall) == expected_recall
    assert pytest.approx(fbeta) == expected_fbeta


def test_train_model():
    X, y = make_classification(n_samples=100,
                               n_features=10,
                               n_classes=2,
                               random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    assert model is not None
