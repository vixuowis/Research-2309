import joblib
import json
import pandas as pd


def predict_environmental_impact(data_file: str, model_file: str = 'model.joblib', config_file: str = 'config.json') -> list:
    """
    Predict the potential negative impact on the environment based on certain factors.

    Args:
        data_file (str): The path to the csv file containing the data to be predicted.
        model_file (str, optional): The path to the pre-trained model file. Defaults to 'model.joblib'.
        config_file (str, optional): The path to the configuration file. Defaults to 'config.json'.

    Returns:
        list: The predicted values.
    """
    model = joblib.load(model_file)
    config = json.load(open(config_file))
    features = config['features']

    data = pd.read_csv(data_file)
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]
    predictions = model.predict(data)
    return predictions.tolist()