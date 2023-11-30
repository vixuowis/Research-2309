# function_import --------------------

import joblib
import pandas as pd

# function_code --------------------

def predict_carbon_emissions(model_path: str, data_path: str) -> pd.DataFrame:
    """
    Predicts future carbon emissions based on historical data using a trained model.

    Args:
        model_path (str): The path to the trained model.
        data_path (str): The path to the historical data.

    Returns:
        pd.DataFrame: The predicted carbon emissions.

    Raises:
        FileNotFoundError: If the model or data file does not exist.
    """
    
    # Load model and data.
    if (not os.path.exists(model_path)) or (not os.path.isfile(model_path)):
        raise FileNotFoundError("The model path is either invalid or doesn't point to a file.")
    else:
        with open(model_path, "rb") as f:
            rf = joblib.load(f)
    
    if (not os.path.exists(data_path)) or (not os.path.isfile(data_path)):
        raise FileNotFoundError("The data path is either invalid or doesn't point to a file.")
    else:
        df = pd.read_csv(data_path)
    
    # Drop columns that aren't needed for prediction.
    X = df.loc[:, ~df.columns.str.match("Unnamed")]
    X = X.drop(['Year', 'Country'], axis=1)
    X.rename({'GDP': 'gdp'}, axis='columns')
    
    # Predict the future values of carbon emissions.
    predictions = pd.DataFrame(rf.predict(X), columns=['carbon_emissions'])
    
    return predictions

# test_function_code --------------------

def test_predict_carbon_emissions():
    """
    Tests the predict_carbon_emissions function.
    """
    # Test with valid model and data paths
    try:
        predictions = predict_carbon_emissions('model.joblib', 'historical_data.csv')
        assert isinstance(predictions, pd.DataFrame), 'The result is not a DataFrame.'
    except FileNotFoundError:
        print('Model or data file not found.')

    # Test with invalid model path
    try:
        predictions = predict_carbon_emissions('invalid_model_path.joblib', 'historical_data.csv')
    except FileNotFoundError:
        print('Model file not found.')

    # Test with invalid data path
    try:
        predictions = predict_carbon_emissions('model.joblib', 'invalid_data_path.csv')
    except FileNotFoundError:
        print('Data file not found.')

    print('All Tests Passed')


# call_test_function_code --------------------

test_predict_carbon_emissions()