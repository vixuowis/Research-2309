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

    X = data.drop(columns=['Electricity'])
    y = data[['Electricity']]

    # split to train/test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # fit random forest model on training data
    rf_regressor = RandomForestRegressor(n_estimators=10, random_state=42)
    rf_regressor.fit(X_train, y_train)
    
    # predict test dataset
    y_pred = rf_regressor.predict(X_test)
    
    # evaluate performance of the prediction
    mse = mean_squared_error(y_test, y_pred)

    return mse

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