# requirements_file --------------------

!pip install -U joblib pandas

# function_import --------------------

import joblib
import pandas as pd

# function_code --------------------

def predict_carbon_emissions(model_file='model.joblib', data_file='data.csv', features=None):
    """
    Load a pre-trained tabular regression model and predict carbon emissions based on input data.

    Args:
        model_file (str): The file path to the saved joblib model.
        data_file (str): The file path to the CSV containing input data.
        features (list of str): List of feature names to be used for prediction.

    Returns:
        pandas.Series: The predicted carbon emissions values for the input data.

    Raises:
        FileNotFoundError: If the model_file or data_file does not exist.
        ValueError: If the features list is empty or None.
    """
    if features is None or not features:
        raise ValueError("Features list cannot be empty or None.")
    try:
        # Load the model using joblib
        model = joblib.load(model_file)
        # Load the data using pandas
        data = pd.read_csv(data_file)
        # Preprocess the dataset to select relevant columns
        data = data[features]
        # Make predictions using the pre-trained model
        return model.predict(data)
    except FileNotFoundError as fnf_error:
        raise FileNotFoundError(f"File not found: {str(fnf_error)}")

# test_function_code --------------------

def test_predict_carbon_emissions():
    print("Testing started.")
    # Using predefined configuration for the test
    config = {'features': ['feature_1', 'feature_2', 'feature_3']}

    # Test case 1: Successful prediction
    print("Testing case [1/1] started.")
    predicted = predict_carbon_emissions(
        model_file='model.joblib',
        data_file='data.csv',
        features=config['features'])
    assert len(predicted) > 0, "Test case [1/1] failed: No predictions returned."
    print("Testing finished.")

# call_test_function_line --------------------

test_predict_carbon_emissions()