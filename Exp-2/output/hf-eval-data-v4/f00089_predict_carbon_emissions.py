# requirements_file --------------------

!pip install -U joblib,pandas

# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def predict_carbon_emissions(model_path, config_path, data_path):
    """
    Predict the carbon emissions of a new line of electric vehicles using a pre-trained model.

    :param model_path: Path to the pre-trained regression model file.
    :param config_path: Path to the configuration file containing feature names.
    :param data_path: Path to the CSV file containing new vehicle data.
    :return: A pandas DataFrame with predicted carbon emissions.
    """
    # Load the regression model
    model = joblib.load(model_path)

    # Load features configuration
    config = json.load(open(config_path))
    features = config['features']

    # Load the dataset and select features
    data = pd.read_csv(data_path)
    data = data[features]

    # Predict emissions
    predictions = model.predict(data)
    return pd.DataFrame(predictions, columns=['Predicted Emissions'])

# test_function_code --------------------

def test_predict_carbon_emissions():
    print("Testing predict_carbon_emissions function.")
    sample_data_path = 'sample_new_vehicle_data.csv'
    sample_model_path = 'sample_model.joblib'
    sample_config_path = 'sample_config.json'

    # Assuming these sample files are correctly setup
    predicted_emissions = predict_carbon_emissions(sample_model_path, sample_config_path, sample_data_path)

    # Test if predictions are returned as a DataFrame
    assert isinstance(predicted_emissions, pd.DataFrame), "Output should be a DataFrame."

    # Test if predictions DataFrame has correct column
    assert 'Predicted Emissions' in predicted_emissions.columns, "DataFrame should have 'Predicted Emissions' as a column."

    # Test if predictions DataFrame is not empty
    assert not predicted_emissions.empty, "Predicted emissions should not be empty."
    print("All tests passed.")

test_predict_carbon_emissions()