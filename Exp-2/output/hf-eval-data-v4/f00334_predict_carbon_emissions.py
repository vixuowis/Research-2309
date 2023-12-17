# requirements_file --------------------

!pip install -U joblib pandas

# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def predict_carbon_emissions(model_file, config_file, data_file):
    """
    Predict carbon emissions for the given dataset using a pre-trained model.

    Parameters:
        model_file (str): The file path for the pre-trained model.
        config_file (str): The file path for the configuration file containing feature information.
        data_file (str): The file path for the new data to predict on.

    Returns:
        list: Predictions of carbon emissions.
    """
    # Load the pre-trained model
    model = joblib.load(model_file)
    # Load the configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    features = config['features']
    # Read new data
    data = pd.read_csv(data_file)
    # Pre-process the data
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]
    # Make predictions
    predictions = model.predict(data)
    return predictions.tolist()

# test_function_code --------------------

def test_predict_carbon_emissions():
    print("Testing predict_carbon_emissions function.")
    predictions = predict_carbon_emissions('model.joblib', 'config.json', 'data.csv')
    # Assuming we have a ground truth for validation
    ground_truth = [100, 150, 200] # This should be obtained from an actual validation set
    assert len(predictions) == len(ground_truth), f"Number of predictions does not match the ground truth"
    assert all(isinstance(pred, float) for pred in predictions), f"Some predictions are not numbers"
    print("All tests passed successfully.")

# Run the test function
test_predict_carbon_emissions()