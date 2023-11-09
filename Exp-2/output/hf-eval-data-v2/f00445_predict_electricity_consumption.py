# function_import --------------------

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd

# function_code --------------------

def predict_electricity_consumption(data):
    """
    This function predicts the electricity consumption of a residential area based on historical data.
    It uses the RandomForestRegressor model from the Scikit-learn library.

    Args:
        data (pd.DataFrame): The historical data. It should include a 'electricity_consumption' column which will be used as the target.

    Returns:
        float: The mean squared error of the model's predictions.
    """
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
    """
    This function tests the 'predict_electricity_consumption' function.
    It uses a small dataset and checks if the output is a float.
    """
    # Create a small test dataset
    data = pd.DataFrame({
        'electricity_consumption': [100, 200, 300, 400, 500],
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1]
    })
    mse = predict_electricity_consumption(data)
    assert isinstance(mse, float), 'The output should be a float.'

# call_test_function_code --------------------

test_predict_electricity_consumption()