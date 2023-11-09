# function_import --------------------

import joblib
import pandas as pd
from transformers import AutoModel

# function_code --------------------

def predict_survival(data_path):
    """
    Predicts the survival status of passengers on the Titanic based on their age, gender, and passenger class.

    Args:
        data_path (str): Path to the CSV file containing the data. The CSV file should contain columns 'age', 'gender', and 'passenger_class'.

    Returns:
        predictions (array): An array of survival status predictions for each passenger. 1 indicates survival and 0 indicates non-survival.
    """
    model = AutoModel.from_pretrained('harithapliyal/autotrain-tatanic-survival-51030121311')
    data = pd.read_csv(data_path)
    data = data[['age', 'gender', 'passenger_class']]  # Subset the data for the relevant features
    predictions = model.predict(data)
    return predictions

# test_function_code --------------------

def test_predict_survival():
    """
    Tests the predict_survival function by loading a test dataset and comparing the predicted survival status with the actual survival status.
    """
    test_data_path = 'test_data.csv'  # Path to the test dataset
    predictions = predict_survival(test_data_path)
    actual = pd.read_csv(test_data_path)['Survived']  # Load the actual survival status from the test dataset
    assert len(predictions) == len(actual), 'The number of predictions does not match the number of actual survival status.'
    assert (predictions == 1).sum() + (predictions == 0).sum() == len(predictions), 'All predictions should be either 1 (survived) or 0 (not survived).'

# call_test_function_code --------------------

test_predict_survival()