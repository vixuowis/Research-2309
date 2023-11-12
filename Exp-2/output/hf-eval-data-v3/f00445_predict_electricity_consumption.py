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
    # Assume data is a Pandas DataFrame, and X is the feature set, y is the target
    X = data.drop('electricity_consumption', axis=1)
    y = data['electricity_consumption']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestRegressor(max_depth=10, n_estimators=50, random_state=59)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
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