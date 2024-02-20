import os
import sys
from fastapi.testclient import TestClient
from starter.main import app

# Add parent directory to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)


client = TestClient(app)
base_url = "http://127.0.0.1:8000"


def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert "Welcome !" in r.text


# Test case 1: Test prediction for a sample input
def test_predict_census_income1():
    # Sample input data
    data = {
            "age": 60,
            "workclass": " Federal-gov",
            "fnlgt": 77516,
            "education": " Some-college",
            "education_num": 13,
            "marital_status": " Married",
            "occupation": " Exec-managerial",
            "relationship": " Husband",
            "race": " White",
            "sex": " Male",
            "capital_gain": 4000,
            "capital_loss": 100,
            "hours_per_week": 40,
            "native_country": " Germany"
        }

    # Send POST request to the /predict endpoint
    response = client.post("/predict", json=data)

    # Check if the request was successful (status code 200)
    assert response.status_code == 200

    # Check if the response contains the predicted label
    assert "prediction" in response.json()
    predicted_label = response.json()["prediction"]
    assert predicted_label in (">50K", "<=50K")


def test_predict_census_income2():
    # Sample input data
    data = {
            "age": 20,
            "workclass": " Federal-gov",
            "fnlgt": 4000,
            "education": " Some-college",
            "education_num": 11,
            "marital_status": " Married",
            "occupation": " Exec-managerial",
            "relationship": " Wife",
            "race": " Black",
            "sex": " Female",
            "capital_gain": 4000,
            "capital_loss": 100,
            "hours_per_week": 40,
            "native_country": " Asian-Pac-Islander"
        }

    # Send POST request to the /predict endpoint
    response = client.post("/predict", json=data)

    # Check if the request was successful (status code 200)
    assert response.status_code == 200

    # Check if the response contains the predicted label
    assert "prediction" in response.json()
    predicted_label = response.json()["prediction"]
    assert predicted_label in (">50K", "<=50K")
