from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd


def predict_electricity_consumption(data):
    """
    This function predicts the electricity consumption of a residential area based on historical data.
    It uses the RandomForestRegressor model from the Scikit-learn library.
    
    Parameters:
    data (DataFrame): The historical data. Assume it's a Pandas DataFrame, and X is the feature set, y is the target.
    
    Returns:
    float: The predicted electricity consumption.
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

    # Train the model
    model = RandomForestRegressor(max_depth=10, n_estimators=50, random_state=59)
    model.fit(X_train, y_train)

    # Predict electricity consumption
    predictions = model.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    return predictions[-1]