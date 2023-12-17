# requirements_file --------------------

!pip install -U tensorflow, keras

# function_import --------------------

import tensorflow as tf
from keras import TFAutoModelForSequenceClassification

# function_code --------------------

def detect_anomalies_in_energy_data(data, model_name='keras-io/timeseries-anomaly-detection'):
    """
    Detect anomalies in energy consumption time series data using a pre-trained model.

    Args:
        data: The time series data where anomalies need to be detected.
        model_name (str): The name of the pre-trained model to use.
    Returns:
        An array indicating the presence of anomalies (1) or normal data (0).

    """
    # Load the pre-trained model
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Assuming data is preprocessed for the model input
    predictions = model.predict(data)

    # Anomaly detected if reconstruction error is above a certain threshold
    anomalies = predictions['reconstruction_error'] > threshold # Define threshold based on model specifics
    return anomalies

# test_function_code --------------------

def test_detect_anomalies_in_energy_data():
    print("Testing detect_anomalies_in_energy_data function.")

    # Sample synthetic energy consumption data
    sample_data = generate_synthetic_data(size=1000)

    # Test case: Checking the anomaly detection output
    print("Testing case [1/1] started.")
    anomalies = detect_anomalies_in_energy_data(sample_data)
    assert anomalies.ndim == 1, "Output should be a 1D array indicating anomalies"
    print("All test cases passed.")