# requirements_file --------------------

!pip install -U scikit-learn pandas numpy

# function_import --------------------

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# function_code --------------------

def predict_electricity_consumption(data):
    """
    Predict the electricity consumption of a residential area using RandomForestRegressor.

    Parameters:
    data (DataFrame): The input historical data with features and target.

    Returns:
    float: The Mean Squared Error of the model's predictions.
    """
    # Assume data is a Pandas DataFrame, and X is the feature set, y is the target
    X = data.drop('electricity_consumption', axis=1)
    y = data['electricity_consumption']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize and train the RandomForestRegressor model
    model = RandomForestRegressor(max_depth=10, n_estimators=50, random_state=59)
    model.fit(X_train, y_train)

    # Predict and calculate Mean Squared Error
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    return mse

# test_function_code --------------------

def test_predict_electricity_consumption():
    print("Testing predict_electricity_consumption started.")
    # Generate synthetic data for testing
    np.random.seed(0)
    columns = ['feature_1', 'feature_2', 'electricity_consumption']
    synthetic_data = pd.DataFrame(np.random.rand(100, 3), columns=columns)
    synthetic_data['electricity_consumption'] *= 100

    # Call the prediction function
    mse = predict_electricity_consumption(synthetic_data)

    # Check if the MSE is within an acceptable range
    assert mse >= 0, f"Test failed: MSE is negative ({mse})"
    print("Testing predict_electricity_consumption finished.")

# Run the test function
if __name__ == '__main__':
    test_predict_electricity_consumption()