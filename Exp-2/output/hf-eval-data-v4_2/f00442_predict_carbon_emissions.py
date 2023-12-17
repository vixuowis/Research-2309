# requirements_file --------------------

!pip install -U json joblib pandas

# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def predict_carbon_emissions(data_path, model_path, config_path):
    """
    Predicts the carbon emissions of facilities based on their data.

    Args:
        data_path (str): The path to the CSV file containing the facilities data.
        model_path (str): The path to the pretrained model file.
        config_path (str): The path to the JSON configuration file specifying the required features.

    Returns:
        DataFrame: A pandas DataFrame containing the predictions of carbon emissions.

    Raises:
        FileNotFoundError: If the data, model, or config file does not exist.
        KeyError: If the required features are not found in the data.

    """
    # Import necessary libraries
    import json
    import joblib
    import pandas as pd

    # Load the pretrained model
    model = joblib.load(model_path)

    # Load the configuration for the features
    with open(config_path) as config_file:
        config = json.load(config_file)
    features = config['features']

    # Load and preprocess the data
    data = pd.read_csv(data_path)
    if not all(feature in data.columns for feature in features):
        missing_features = [feature for feature in features if feature not in data.columns]
        raise KeyError(f'Required features {missing_features} not found in data.')
    data = data[features]
    
    # Rename the data columns to match the naming convention
    data.columns = ['feat_' + str(col) for col in data.columns]

    # Predict the carbon emissions and return the results
    predictions = model.predict(data)
    return pd.DataFrame(predictions, columns=['Carbon_Emissions'])

# test_function_code --------------------

def test_predict_carbon_emissions():
    print("Testing started.")

    # Test with sample data path, model path, and config path
    test_data_path = 'sample_data.csv'
    test_model_path = 'test_model.joblib'
    test_config_path = 'test_config.json'

    # Test case 1: Verify the function raises a FileNotFoundError when files are missing
    print("Testing case [1/3] started.")
    try:
        predict_carbon_emissions('non_existent_data.csv', test_model_path, test_config_path)
        assert False, "Test case [1/3] failed: FileNotFoundError was not raised."
    except FileNotFoundError:
        pass

    # Test case 2: Verify the function raises a KeyError if required features are missing
    print("Testing case [2/3] started.")
    try:
        predict_carbon_emissions(test_data_path, test_model_path, 'non_existent_config.json')
        assert False, "Test case [2/3] failed: KeyError was not raised."
    except KeyError:
        pass

    # Test case 3: Verify the function returns a DataFrame with correct predictions
    print("Testing case [3/3] started.")
    result = predict_carbon_emissions(test_data_path, test_model_path, test_config_path)
    assert not result.empty, "Test case [3/3] failed: The result should not be empty."
    assert 'Carbon_Emissions' in result.columns, "Test case [3/3] failed: The column 'Carbon_Emissions' should be present in the result."

    print("Testing finished.")

# call_test_function_line --------------------

test_predict_carbon_emissions()