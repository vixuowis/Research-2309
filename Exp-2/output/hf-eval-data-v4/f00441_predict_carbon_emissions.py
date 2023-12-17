# requirements_file --------------------

!pip install -U joblib pandas numpy

# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def predict_carbon_emissions(material_csv, model_path='model.joblib', config_path='config.json'):
    """
    Predict the carbon footprint of a construction project based on material consumption data.

    Parameters:
        material_csv (str): The path to the CSV file containing material consumption data.
        model_path (str): The path to the pre-trained model file.
        config_path (str): The path to the configuration file detailing the features.

    Returns:
        array: Predicted carbon emission values.
    """
    # Load the pre-trained model
    model = joblib.load(model_path)

    # Load the features configuration
    config = json.load(open(config_path))
    features = config['features']

    # Read the material consumption data
    data = pd.read_csv(material_csv)

    # Select and rename columns as per the model's expected format
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]

    # Make predictions
    predictions = model.predict(data)
    return predictions

# test_function_code --------------------

def test_predict_carbon_emissions():
    print("Testing started.")
    # Note that this is a hypothetical test; actual file paths would be different.
    sample_data = 'sample_material_consumption.csv'  # Hypothetical sample data file

    # Test case 1: Check if predictions are returned as an array
    print("Testing case [1/1] started.")
    predictions = predict_carbon_emissions(sample_data)
    assert isinstance(predictions, np.ndarray), f"Test case [1/1] failed: Expected predictions to be numpy array, got {type(predictions)}"
    print("Testing finished.")

# Run the test function
test_predict_carbon_emissions()