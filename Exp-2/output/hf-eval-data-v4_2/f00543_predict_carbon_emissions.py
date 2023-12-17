# requirements_file --------------------

!pip install -U json joblib pandas

# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def predict_carbon_emissions(input_data_file):
    """
    Predict the carbon emissions of power plants given their characteristics.

    Args:
        input_data_file (str): The file path to the CSV file containing the power plant characteristics.

    Returns:
        List[float]: A list of predicted carbon emission values for the input samples.

    Raises:
        FileNotFoundError: If the input data file or configuration file is not found.
        KeyError: If the input data is missing required features.
    """
    # Load the trained model
    model = joblib.load('model.joblib')
    
    # Load the configuration file
    config = json.load(open('config.json'))
    features = config['features']
    
    # Process the input data
    data = pd.read_csv(input_data_file)
    if set(features).issubset(data.columns):
        data = data[features]
    else:
        raise KeyError('Input data is missing some required features.')
    data.columns = ['feat_' + str(col) for col in data.columns]
    
    # Make predictions
    predictions = model.predict(data)
    return predictions.tolist()

# test_function_code --------------------

def test_predict_carbon_emissions():
    print("Testing started.")
    # Test case 1: Valid input file
    print("Testing case [1/3] started.")
    predictions = predict_carbon_emissions('valid_data.csv')
    assert isinstance(predictions, list) and all(isinstance(x, float) for x in predictions), f"Test case [1/3] failed: predictions must be a list of float values."

    # Test case 2: Non-existent input file
    print("Testing case [2/3] started.")
    try:
        predict_carbon_emissions('nonexistent_data.csv')
        assert False, "Test case [2/3] failed: FileNotFoundError was not raised."
    except FileNotFoundError:
        pass

    # Test case 3: Input file missing required features
    print("Testing case [3/3] started.")
    try:
        predict_carbon_emissions('incomplete_data.csv')
        assert False, "Test case [3/3] failed: KeyError was not raised."
    except KeyError:
        pass
    print("Testing finished.")

# call_test_function_line --------------------

test_predict_carbon_emissions()