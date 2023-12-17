# requirements_file --------------------

import subprocess

requirements = ["joblib", "pandas"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import joblib
import pandas as pd

# function_code --------------------

def model_predict_emissions(input_data):
    """
    Predicts carbon emissions category for a building given its features.

    Args:
        input_data (DataFrame): A pandas DataFrame containing the building features.

    Returns:
        ndarray: A numpy array containing the predicted carbon emissions categories.

    Raises:
        FileNotFoundError: If 'model.joblib' is not found in the current directory.
        KeyError: If required feature columns are missing in the input data.
    """
    # Load the pre-trained model
    model = joblib.load('model.joblib')
    # Check if input data has the required columns specified in 'config.json'
    config = json.load(open('config.json'))
    required_columns = config['features']
    if not all(column in input_data for column in required_columns):
        raise KeyError('Input data must contain the following columns: {}'.format(required_columns))
    # Preparing the input data for prediction
    input_df = pd.DataFrame(input_data)
    input_df.columns = ['feat_' + str(col) for col in input_df.columns]
    # Predicting carbon emissions category
    predictions = model.predict(input_df)
    return predictions

# test_function_code --------------------

def test_model_predict_emissions():
    print("Testing started.")
    # Load sample input data
    sample_input = pd.read_csv('sample_data.csv')

    # Test case 1: Correctly formatted input data
    print("Testing case [1/3] started.")
    predictions = model_predict_emissions(sample_input.head())
    assert len(predictions) > 0, "Test case [1/3] failed: No predictions returned."

    # Test case 2: Missing columns in input data
    print("Testing case [2/3] started.")
    incomplete_data = sample_input.drop(columns=sample_input.columns[0])
    try:
        model_predict_emissions(incomplete_data)
        assert False, "Test case [2/3] failed: KeyError for missing columns not raised."
    except KeyError:
        pass

    # Test case 3: Model file not found
    print("Testing case [3/3] started.")
    try:
        model_predict_emissions(sample_input.head())
        assert False, "Test case [3/3] failed: FileNotFoundError not raised."
    except FileNotFoundError:
        pass
    print("Testing finished.")

# call_test_function_line --------------------

test_model_predict_emissions()