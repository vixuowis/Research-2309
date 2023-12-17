# requirements_file --------------------

!pip install -U pandas transformers

# function_import --------------------

import pandas as pd
from transformers import AutoModel

# function_code --------------------

def predict_titanic_survival(input_csv_path):
    """
    Predict the survival status of passengers on the Titanic.

    Args:
        input_csv_path (str): The file path to the CSV containing passenger data.

    Returns:
        pandas.Series: A series of predictions where 1 represents survival and 0 represents non-survival.

    Raises:
        FileNotFoundError: If the input CSV file does not exist.
        ValueError: If the input data is not in the correct format.
    """
    try:
        data = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        raise FileNotFoundError('The input CSV file was not found.')
    
    if not all(x in data.columns for x in ['age', 'gender', 'passenger_class']):
        raise ValueError('Input CSV must contain age, gender, and passenger_class columns.')
    
    data = data[['age', 'gender', 'passenger_class']]
    model = AutoModel.from_pretrained('harithapliyal/autotrain-tatanic-survival-51030121311')
    predictions = model.predict(data)
    return predictions

# test_function_code --------------------

def test_predict_titanic_survival():
    print("Testing started.")
    
    # Test case 1: Correct CSV input
    print("Testing case [1/2] started.")
    try:
        predictions = predict_titanic_survival('test_data.csv')
        assert isinstance(predictions, pd.Series), 'Predictions should be a pandas Series.'
    except Exception as e:
        assert False, f"Test case [1/2] failed: {e}"
    
    # Test case 2: Non-existent CSV file
    print("Testing case [2/2] started.")
    try:
        predict_titanic_survival('non_existent.csv')
        assert False, 'Test case [2/2] should have raised FileNotFoundError.'
    except FileNotFoundError:
        pass # Expected outcome
    
    print("Testing finished.")

# call_test_function_line --------------------

test_predict_titanic_survival()