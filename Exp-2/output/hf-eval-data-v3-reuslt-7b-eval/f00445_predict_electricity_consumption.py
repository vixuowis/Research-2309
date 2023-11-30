# function_import --------------------

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# function_code --------------------

def predict_electricity_consumption(data):
    '''
    Predict the electricity consumption of a residential area based on historical data using RandomForestRegressor.

    Args:
        data (pd.DataFrame): The historical data with features and target.

    Returns:
        float: The mean squared error of the prediction.
    '''
    
    # Split the data into X (features) and y (target). 
    X = data.drop(['electricity_consumption'], axis=1)
    y = data['electricity_consumption']
    
    # Split the dataset for training and testing.
    trainX, testX, trainy, testy = train_test_split(X, y, test_size = 0.25, random_state=42)
        
    # Standardize features by removing the mean and scaling to unit variance.
    sc = StandardScaler()
    sc.fit(trainX)
    
    # Apply standardization by transforming trainX and testX.
    trainXstd = sc.transform(trainX)
    testXstd = sc.transform(testX)
        
    # Define the regressor. 
    rfr = RandomForestRegressor()
    
    # Fit the model using training data.
    rfr.fit(trainXstd, trainy)
                
    # Predict electricity consumption on testing data. 
    predict_y = rfr.predict(testXstd)
    
    return mean_squared_error(predict_y, testy)

# test_function_code --------------------

def test_predict_electricity_consumption():
    '''
    Test the function predict_electricity_consumption.
    '''
    # Create a random dataset for testing
    data = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
    data['electricity_consumption'] = np.random.randint(0,100,size=(100, 1))

    # Call the function with the test dataset
    mse = predict_electricity_consumption(data)

    # Since we are using random data, we can't predict the exact output.
    # So, we just check if the output is a float number.
    assert isinstance(mse, float), 'The result should be a float.'

    print('All Tests Passed')


# call_test_function_code --------------------

test_predict_electricity_consumption()