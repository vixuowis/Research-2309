# function_import --------------------

import joblib
import pandas as pd

# function_code --------------------

def predict_customer_purchase(model_path: str, data_path: str) -> pd.DataFrame:
    """
    Predicts which customers will make a purchase based on their browsing behavior.

    Args:
        model_path (str): The path to the trained model.
        data_path (str): The path to the customer browsing data.

    Returns:
        pd.DataFrame: The predictions of the model.

    Raises:
        FileNotFoundError: If the model or data file does not exist.
    """
    try:
        with open(model_path, "rb") as fp:
            clf = joblib.load(fp)

        X = pd.read_csv(data_path).drop("customer", axis=1)
        
        y = clf.predict(X)
    except FileNotFoundError as e:
        raise e
    else:
        return pd.DataFrame({"y":y})

# test_function_code --------------------

def test_predict_customer_purchase():
    """Tests the predict_customer_purchase function."""
    model_path = 'test_model.joblib'
    data_path = 'test_data.csv'
    try:
        predictions = predict_customer_purchase(model_path, data_path)
    except FileNotFoundError:
        print('Test model or data file does not exist.')
    else:
        assert isinstance(predictions, pd.DataFrame), 'The result is not a DataFrame.'
        print('All Tests Passed')


# call_test_function_code --------------------

test_predict_customer_purchase()