import numpy as np
import pandas as pd
from ml.model import inference , compute_model_metrics 
from ml.data import process_data
import joblib

def compute_performance_on_slices(model ,data, encoder , lb,  feature_name):
    """
    Computes performance metrics for model slices based on a categorical variable.

    Inputs
    ------
    model .
    data
    feature_name : str
        Name of the categorical feature.

    Returns
    -------
    slice_metrics : dict
        Dictionary containing performance metrics for each slice.
    """
    # Get unique values of the categorical feature
    print(data.columns)
    unique_values = np.unique(data[feature_name])
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
    
    
    # Initialize dictionary to store performance metrics for each slice
    slice_metrics = {}
    X_processed, y_processed, _, _ = process_data(X=data  ,
                                  categorical_features = cat_features,
                                  label = 'salary' ,
                                  encoder=encoder ,
                                  lb = lb,
                                  training = False)
    # Iterate through unique values
    for value in unique_values:
        # Select data points where the feature equals the current value
        indices = data[feature_name] == value
        
        X_slice = X_processed[indices]
        y_slice = y_processed[indices]

        # Compute model metrics for the slice
        y_pred = model.predict(X_slice)  # Assuming 'model' is trained
        precision, recall, f1 = compute_model_metrics(y_slice, y_pred)

        # Store metrics in dictionary
        slice_metrics[value] = {'precision': precision, 'recall': recall, 'f1': f1}

    return slice_metrics

if __name__ == "__main__":
    model = joblib.load("..//model//trained_model.pkl")
    encoder = joblib.load("..//model//encoder.pkl")
    lb = joblib.load("..//model//lb.pkl")

    data = pd.read_csv("..//data//census.csv")
    data.rename(columns=lambda x: x.replace(' ', ''), inplace=True)
    
    print(compute_performance_on_slices(model = model ,data = data, encoder = encoder , lb = lb, feature_name= 'education'))
