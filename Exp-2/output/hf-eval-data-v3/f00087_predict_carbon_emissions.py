# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def predict_carbon_emissions(data_file: str, model_file: str, config_file: str) -> pd.DataFrame:
    """
    Predict carbon emissions based on the given features of the compound.

    Args:
        data_file (str): The path to the input data file in CSV format.
        model_file (str): The path to the pretrained model file.
        config_file (str): The path to the configuration file in JSON format.

    Returns:
        pd.DataFrame: The predicted carbon emissions.

    Raises:
        FileNotFoundError: If any of the input files does not exist.
    """
    # Load the pretrained model
    model = joblib.load(model_file)

    # Load the configuration
    config = json.load(open(config_file))

    # Load the input data
    data = pd.read_csv(data_file)

    # Select the important features
    features = config['features']
    data = data[features]

    # Rename the columns
    data.columns = ['feat_' + str(col) for col in data.columns]

    # Make predictions
    predictions = model.predict(data)

    return predictions

# test_function_code --------------------

def test_predict_carbon_emissions():
    """Tests the predict_carbon_emissions function."""
    # Define the input files
    data_file = 'test_data.csv'
    model_file = 'test_model.joblib'
    config_file = 'test_config.json'

    # Call the function
    predictions = predict_carbon_emissions(data_file, model_file, config_file)

    # Check the output
    assert predictions is not None, 'The predictions should not be None.'
    assert isinstance(predictions, pd.DataFrame), 'The output should be a pandas DataFrame.'
    assert not predictions.empty, 'The DataFrame should not be empty.'

    print('All Tests Passed')

# call_test_function_code --------------------

if __name__ == '__main__':
    test_predict_carbon_emissions()