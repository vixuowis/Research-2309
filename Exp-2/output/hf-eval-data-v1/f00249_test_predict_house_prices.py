import pandas as pd

# Test function for predict_house_prices
# The function loads a test dataset and uses the predict_house_prices function to predict the house prices
# It then checks if the predictions are not null

def test_predict_house_prices():
    # Load the test dataset
    test_data = pd.read_csv('test_data.csv')
    
    # Use the predict_house_prices function to predict the house prices
    predictions = predict_house_prices(test_data)
    
    # Check if the predictions are not null
    assert predictions is not None
    
    # Check if the predictions are not empty
    assert len(predictions) > 0
    
    # Check if the predictions are numbers
    assert all(isinstance(p, (int, float)) for p in predictions)