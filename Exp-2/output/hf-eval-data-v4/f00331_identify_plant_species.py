# requirements_file --------------------

!pip install -U joblib,pandas

# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def identify_plant_species(data_file, model_file='model.joblib', config_file='config.json'):
    """
    Identify the species of plants among Iris Setosa, Iris Versicolor, and Iris Virginica.

    :param data_file: CSV file containing the dataset of plants to classify.
    :param model_file: File path to the pre-trained KNN model.
    :param config_file: File path to the configuration file.
    :return: List of predicted plant species.
    """
    # Load the pre-trained KNN model
    model = joblib.load(model_file)

    # Load the configuration file
    with open(config_file, 'r') as f:
        config = json.load(f)
    features = config['features']

    # Read the dataset
    data = pd.read_csv(data_file)
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]

    # Perform predictions
    predictions = model.predict(data)
    return predictions.tolist()

# test_function_code --------------------

def test_identify_plant_species():
    print('Testing identify_plant_species function...')
    predictions = identify_plant_species('test_data.csv')
    expected_species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    # Check if the predictions are in the expected species list
    assert all(species in expected_species for species in predictions), 'Some predictions are not among the expected species.'
    print('Testing completed successfully.')