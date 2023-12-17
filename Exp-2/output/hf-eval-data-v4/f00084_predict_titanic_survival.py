# requirements_file --------------------

!pip install -U pandas transformers

# function_import --------------------

import pandas as pd
from transformers import AutoModel

# function_code --------------------

def predict_titanic_survival(data_csv_path):
    """
    Predict the survival status of passengers on the Titanic.

    Parameters:
    data_csv_path (str): The path to the CSV file containing passenger data with 'age', 'gender',
                        and 'passenger class' columns.

    Returns:
    list: Predictions indicating survival status (1 for survived, 0 for did not survive).
    """
    # Load the pre-trained model from Hugging Face's Model Hub
    model = AutoModel.from_pretrained('harithapliyal/autotrain-tatanic-survival-51030121311')

    # Load the input data from CSV
    data = pd.read_csv(data_csv_path)
    # Subset the data for the relevant features
    data = data[['age', 'gender', 'passenger_class']]
    # Predict survival status
    predictions = model.predict(data)
    return predictions

# test_function_code --------------------

def test_predict_titanic_survival():
    print("Testing predict_titanic_survival function.")

    # Prepare a sample CSV file path (this needs to be replaced with an actual file path)
    sample_csv = 'test_data.csv'

    # Test case 1: Check if the function returns a list
    print("Test case 1: Checking return type.")
    predictions = predict_titanic_survival(sample_csv)
    assert isinstance(predictions, list), "Function should return a list of predictions."

    # Test case 2: Check if predictions are binary
    print("Test case 2: Checking prediction values.")
    assert all(p in [0, 1] for p in predictions), "Predictions should be binary (0 or 1)."

    # Test case 3: Check if number of predictions matches number of rows in CSV
    print("Test case 3: Checking number of predictions.")
    data = pd.read_csv(sample_csv)
    assert len(predictions) == len(data), "Number of predictions should match number of rows in input CSV."

    print("All tests passed.")