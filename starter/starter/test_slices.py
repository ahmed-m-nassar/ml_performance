import numpy as np
import pandas as pd

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..' , '..'))
sys.path.insert(0, root_dir)

from starter.starter.ml.model import inference , compute_model_metrics 
from starter.starter.ml.data import process_data
import joblib
import os

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
        #print(X_slice)
        y_slice = y_processed[indices]

        # Compute model metrics for the slice
        y_pred = model.predict(X_slice)  # Assuming 'model' is trained
        precision, recall, f1 = compute_model_metrics(y_slice, y_pred)

        # Store metrics in dictionary
        slice_metrics[value] = {'precision': precision, 'recall': recall, 'f1': f1}

    return slice_metrics

if __name__ == "__main__":
    current_directory = os.path.dirname(__file__)
    
    data_path = os.path.join(current_directory, '..', 'data', 'census.csv')
    model_path = os.path.join(current_directory, '..', 'model', 'trained_model.pkl')
    encoder_path = os.path.join(current_directory, '..', 'model', 'encoder.pkl')
    lb_path = os.path.join(current_directory, '..', 'model', 'lb.pkl')
    

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    lb = joblib.load(lb_path)

    data = pd.read_csv(data_path)
    data.rename(columns=lambda x: x.replace(' ', ''), inplace=True)
    
    #print(data.info())
    np.set_printoptions(threshold=np.inf)

    result = compute_performance_on_slices(model = model ,data = data, encoder = encoder , lb = lb, feature_name= 'workclass')
    with open("slice_output.txt", 'w') as f:
        for key, value in result.items():
            f.write(f"{key}: {value}\n")
