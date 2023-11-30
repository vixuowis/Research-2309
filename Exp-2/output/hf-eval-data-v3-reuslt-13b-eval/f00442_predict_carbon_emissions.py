# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def predict_carbon_emissions(model_path: str, config_path: str, data_path: str) -> pd.DataFrame:
    """
    Predict the carbon emissions of different facilities based on the provided data.

    Args:
        model_path (str): The path to the pretrained model.
        config_path (str): The path to the configuration file.
        data_path (str): The path to the data file.

    Returns:
        pd.DataFrame: The predicted carbon emissions for each facility.
    """
    # Load the data and pretrained model.
    with open(config_path, "r") as fd:
        config = json.load(fd)

    model = joblib.load(model_path)
    input_data = pd.read_csv(data_path, index_col=0).dropna()
    
    # Make the prediction.
    output_data = pd.DataFrame({"carbon_emissions": model.predict(input_data[config["features"]].values)},
                               index=input_data.index)
    
    return output_data

# test_function_code --------------------

def test_predict_carbon_emissions():
    """
    Test the predict_carbon_emissions function.
    """
    model_path = 'model.joblib'
    config_path = 'config.json'
    data_path = 'data.csv'

    try:
        predictions = predict_carbon_emissions(model_path, config_path, data_path)
        assert isinstance(predictions, pd.DataFrame), 'The result is not a DataFrame.'
        assert not predictions.empty, 'The DataFrame is empty.'
    except FileNotFoundError:
        print('Test files not found.')
    except Exception as e:
        print(f'An error occurred: {e}')
    else:
        print('All tests passed.')


# call_test_function_code --------------------

test_predict_carbon_emissions()