# requirements_file --------------------

!pip install -U pandas joblib

# function_import --------------------

import pandas as pd
from transformers import AutoModelForSequenceClassification
from joblib import load

# function_code --------------------

def estimate_co2_emissions(engine_size, transmission_type, miles_traveled):
    """
    Estimate CO2 emissions based on the engine size, 
    transmission type, and miles traveled using a pre-trained model.

    :param engine_size: float, the size of the vehicle's engine in liters
    :param transmission_type: str, type of the vehicle's transmission ('automatic' or 'manual')
    :param miles_traveled: int, the number of miles the vehicle has traveled
    :return: float, estimated CO2 emissions
    """

    # Load the pre-trained model
    model = load('model.joblib')  

    # Prepare the input data
    data = pd.DataFrame({
        'engine_size': [engine_size],
        'transmission_type': [transmission_type],
        'miles_traveled': [miles_traveled]
    })

    # Predict the CO2 emissions
    predictions = model.predict(data)

    # Return the prediction
    return predictions[0]

# test_function_code --------------------

def test_estimate_co2_emissions():
    print("Testing estimate_co2_emissions() started.")
    
    # Test case 1: Small engine, automatic, low mileage
    print("Testing case [1/3] started.")
    result = estimate_co2_emissions(1.2, 'automatic', 5000)
    assert isinstance(result, float), f"Test case [1/3] failed: Result should be a float, got {type(result)} instead."
    
    # Test case 2: Large engine, manual, high mileage
    print("Testing case [2/3] started.")
    result = estimate_co2_emissions(5.0, 'manual', 50000)
    assert isinstance(result, float), f"Test case [2/3] failed: Result should be a float, got {type(result)} instead."
    
    # Test case 3: Medium engine, automatic, medium mileage
    print("Testing case [3/3] started.")
    result = estimate_co2_emissions(2.5, 'automatic', 20000)
    assert isinstance(result, float), f"Test case [3/3] failed: Result should be a float, got {type(result)} instead."
    
    print("Testing estimate_co2_emissions() finished.")

test_estimate_co2_emissions()