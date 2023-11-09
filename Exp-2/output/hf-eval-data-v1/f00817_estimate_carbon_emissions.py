import json
import joblib
import pandas as pd


def estimate_carbon_emissions(data_file: str, model_file: str = 'model.joblib', config_file: str = 'config.json') -> pd.DataFrame:
    """
    Estimate the carbon emissions of a specific device.

    Args:
        data_file (str): The path to the CSV data file with the device's idle power, standby power, and active power.
        model_file (str, optional): The path to the pre-trained model file. Defaults to 'model.joblib'.
        config_file (str, optional): The path to the configuration file containing feature information. Defaults to 'config.json'.

    Returns:
        pd.DataFrame: The estimated carbon emissions for each row in the input data.
    """
    model = joblib.load(model_file)
    config = json.load(open(config_file))
    features = config['features']
    data = pd.read_csv(data_file)
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]
    predictions = model.predict(data)
    return predictions