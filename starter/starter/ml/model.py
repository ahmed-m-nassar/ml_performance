from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV    
import numpy as np
# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Define the hyperparameters to tune
    param_grid = {
        'n_estimators': [200],  # Number of trees in the forest
        'max_depth': [10, 20],       # Maximum depth of the trees
        'min_samples_split': [2],   # Minimum number of samples required to split a node
    }
    
    # Create a RandomForestClassifier
    rf = RandomForestClassifier()

    # Grid search with cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_
    
    return best_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    
    return model.predict(X)


def compute_performance_on_slices(model ,X, y, feature_name):
    """
    Computes performance metrics for model slices based on a categorical variable.

    Inputs
    ------
    model .
    X : np.array
        Feature matrix.
    y : np.array
        Labels.
    feature_name : str
        Name of the categorical feature.

    Returns
    -------
    slice_metrics : dict
        Dictionary containing performance metrics for each slice.
    """
    # Get unique values of the categorical feature
    unique_values = np.unique(X[:, feature_name])

    # Initialize dictionary to store performance metrics for each slice
    slice_metrics = {}

    # Iterate through unique values
    for value in unique_values:
        # Select data points where the feature equals the current value
        indices = X[:, feature_name] == value
        X_slice = X[indices]
        y_slice = y[indices]

        # Compute model metrics for the slice
        y_pred = model.predict(X_slice)  # Assuming 'model' is trained
        precision, recall, f1 = compute_model_metrics(y_slice, y_pred)

        # Store metrics in dictionary
        slice_metrics[value] = {'precision': precision, 'recall': recall, 'f1': f1}

    return slice_metrics