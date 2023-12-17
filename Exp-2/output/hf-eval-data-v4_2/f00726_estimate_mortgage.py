# requirements_file --------------------

!pip install -U joblib pandas

# function_import --------------------

import joblib
import pandas as pd

# function_code --------------------

def estimate_mortgage(data, model_path='model.joblib', config_path='config.json'):
    """
    Estimate the mortgage for housing based on its features.

    Args:
        data (pandas.DataFrame): The dataframe containing the housing features.
        model_path (str): The path to the pre-trained model file.
        config_path (str): The path to the model configuration file.

    Returns:
        pandas.Series: A series of predicted mortgage estimates.

    Raises:
        FileNotFoundError: An error occurred accessing the model or config file.
        ValueError: An error occurred processing the input data.
    """
    try:
        # Load the pre-trained model
        model = joblib.load(model_path)
        # Load the configuration for required features
        config = json.load(open(config_path))
        filtered_columns = config['features']
        # Filter and adjust the data for the model
        if not all(item in data.columns for item in filtered_columns):
            raise ValueError('Input data must contain all required features.')
        data = data[filtered_columns]
        data.columns = ['feat_' + str(col) for col in data.columns]
        # Make predictions
        predictions = model.predict(data)
        return pd.Series(predictions)
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Failed to load files: {e}')

# test_function_code --------------------

def test_estimate_mortgage():
    print("Testing started.")
    # Set up the test data and configuration
    test_data = pd.DataFrame(...)
    # Define paths to model and config
    model_path = 'model.joblib'
    config_path = 'config.json'

    # Testing case 1
    print("Testing case [1/1] started.")
    predictions = estimate_mortgage(test_data, model_path, config_path)
    assert isinstance(predictions, pd.Series), "Test case [1/1] failed: The result should be a pandas Series."
    print("Testing finished.")

# call_test_function_line --------------------

test_estimate_mortgage()