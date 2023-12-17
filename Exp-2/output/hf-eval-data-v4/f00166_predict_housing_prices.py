# requirements_file --------------------

!pip install -U joblib, pandas, numpy

# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def predict_housing_prices(data_path, config_path, model_path):
    """
    Predict housing prices in the US based on the provided data.
    
    Parameters:
    - data_path: str, path to the CSV file containing housing feature data.
    - config_path: str, path to the JSON configuration file specifying which features to use.
    - model_path: str, path to the pre-trained model file to be used for prediction.
    
    Returns:
    - predictions: numpy.ndarray, the predicted housing prices.
    """
    # Load the model
    model = joblib.load(model_path)
    
    # Load the dataset
    data = pd.read_csv(data_path)
    
    # Load configuration and extract features
    config = json.load(open(config_path))
    features = config['features']
    
    # Filter and rename the columns in the dataset
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]
    
    # Predict the housing prices
    predictions = model.predict(data)
    
    return predictions

# test_function_code --------------------

