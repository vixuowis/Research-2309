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
    
    # Load trained model
    try: 
        trained_model = joblib.load(model_path)
    except FileNotFoundError as err: 
        raise FileNotFoundError("The model file could not be found.") from err
        
    # Load historical data
    try: 
        historical_data = pd.read_csv(data_path)
    except FileNotFoundError as err: 
        raise FileNotFoundError("The data file could not be found.") from err    
    
    # Make prediction using the model and historical data
    predicted_carbon_emissions = trained_model.predict(historical_data)
        
    return pd.DataFrame({"predicted_carbon_emissions": predicted_carbon_emissions})

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