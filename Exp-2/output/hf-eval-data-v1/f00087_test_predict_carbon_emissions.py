import pandas as pd

# Function to test the predict_carbon_emissions function
# The function loads a test dataset and selects a sample
# It then calls the predict_carbon_emissions function with the sample
# Finally, it checks the output of the function using assert

def test_predict_carbon_emissions():
    # Load the test dataset
    data = pd.read_csv('kochetkovIT/autotrain-data-ironhack.csv')
    # Select a sample from the dataset
    sample = data.sample(n=10)
    # Call the predict_carbon_emissions function with the sample
    predictions = predict_carbon_emissions(sample)
    # Check the output of the function
    assert len(predictions) == 10, 'The number of predictions should be equal to the number of samples'
    assert all(isinstance(prediction, float) for prediction in predictions), 'All predictions should be floats'

test_predict_carbon_emissions()