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
    # Extract X_train, y_train from DataFrame
    train = data.drop(columns=['Electricity'])
    X_train = train.loc[0 : 128]
    y_train = data.loc[0 : 128, 'Electricity']
    
    # Extract X_test and y_test from DataFrame (the next day)
    X_test = train.iloc[129].values.reshape(1, -1)
    y_test = data.iloc[129, -1]
    
    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    
    # Train the model
    regressor = RandomForestRegressor(n_estimators=10, random_state=42)
    regressor.fit(X_train_scaled, y_train)
    
    # Predict and calculate mean squared error
    prediction = regressor.predict(X_test_scaled)[0]
    MSE = mean_squared_error(y_test, prediction)
    
    return MSE


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