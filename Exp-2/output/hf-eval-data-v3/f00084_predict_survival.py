# function_import --------------------

import joblib
import pandas as pd
from transformers import AutoModel
import numpy as np

# function_code --------------------

def predict_survival(data_path):
    """
    Predicts the survival status of passengers on the Titanic based on their age, gender, and passenger class.

    Args:
        data_path (str): Path to the CSV file containing the data. The CSV file should contain columns such as 'age', 'gender', and 'passenger class'.

    Returns:
        predictions (numpy array): Predicted survival status for each passenger.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        KeyError: If the required columns are not present in the data.
    """
    model = AutoModel.from_pretrained('harithapliyal/autotrain-tatanic-survival-51030121311')
    data = pd.read_csv(data_path)
    data = data[['age', 'gender', 'passenger_class']]
    predictions = model.predict(data)
    return predictions

# test_function_code --------------------

def test_predict_survival():
    """Tests the predict_survival function."""
    test_data_path = 'test_data.csv'
    predictions = predict_survival(test_data_path)
    assert isinstance(predictions, np.ndarray), 'The result is not a numpy array.'
    assert predictions.shape[0] == pd.read_csv(test_data_path).shape[0], 'The number of predictions does not match the number of passengers.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_predict_survival()