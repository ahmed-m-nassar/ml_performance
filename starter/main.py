from fastapi import FastAPI
from pydantic import BaseModel , Field
import joblib
import numpy as np
import os
import sys
import pandas as pd

from starter.starter.ml.model import inference
from starter.starter.ml.data import process_data

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

# Instantiate the app.
app = FastAPI()
current_directory = os.path.dirname(__file__)
model_path = os.path.join(current_directory, 'model', 'trained_model.pkl')
encoder_path = os.path.join(current_directory, 'model', 'encoder.pkl')
model = joblib.load(model_path)
encoder = joblib.load(encoder_path)


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return "Welcome !"


class CensusDataInput(BaseModel):
    age: int = Field(examples=[60])
    workclass: str = Field(examples=[" State-gov"])
    fnlgt: int = Field(examples=[77516])
    education: str = Field(examples=[" Bachelors"])
    education_num: int = Field(examples=[13])
    marital_status: str = Field(examples=[" Never-married"])
    occupation: str = Field(examples=[" Adm-clerical"])
    relationship: str = Field(examples=[" Not-in-family"])
    race: str = Field(examples=[" White"])
    sex: str = Field(examples=["Male"])
    capital_gain: int = Field(examples=[2174])
    capital_loss: int = Field(examples=[1000])
    hours_per_week: int = Field(examples=[20])
    native_country: str = Field(examples=[" United-States"])


@app.post("/predict")
def predict_census_income(data: CensusDataInput):
    # Convert input data to numpy array
    input_data = np.array([[data.age, data.workclass, data.fnlgt,
                            data.education, data.education_num,
                            data.marital_status, data.occupation,
                            data.relationship, data.race,
                            data.sex, data.capital_gain,
                            data.capital_loss, data.hours_per_week,
                            data.native_country]])

    # Define column names
    column_names = ["age", "workclass", "fnlgt",
                    "education", "education-num",
                    "marital-status", "occupation",
                    "relationship", "race", "sex",
                    "capital-gain", "capital-loss",
                    "hours-per-week", "native-country"]

    # Convert input_data to a DataFrame
    df = pd.DataFrame(input_data, columns=column_names)

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
    # Convert numerical columns to numeric data types
    numeric_columns = ["age", "fnlgt", "education-num",
                       "capital-gain", "capital-loss",
                       "hours-per-week"]
    df[numeric_columns] = df[numeric_columns].astype(int)

    # Make prediction
    X_test, _, _, _ = process_data(
        df, categorical_features=cat_features, training=False, encoder=encoder
    )

    prediction = inference(model, X_test)

    # Convert prediction to string label
    predicted_label = ">50K" if prediction[0] else "<=50K"

    return {"prediction": predicted_label}
