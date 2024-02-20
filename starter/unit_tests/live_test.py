import os
import sys
import requests

# Add parent directory to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)


# Test case 1: Test prediction for a sample input
if __name__ == "__main__":
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
    url = "https://ml-performance.onrender.com/predict"
    response = requests.post(url, json=data)
    # Send POST request to the /predict endpoint
    # Check if the request was successful (status code 200)
    print("response code : " + str(response.status_code))
    print("response json : " + str(response.json()))
