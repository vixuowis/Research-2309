from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def predict_electricity_consumption(X, y):
    """
    This function uses RandomForestRegressor to predict electricity consumption.

    Args:
        X (numpy array or pandas DataFrame): The input features for the model.
        y (numpy array or pandas Series): The target variable for the model.

    Returns:
        predictions (numpy array): The predicted electricity consumption.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=59)

    # Create an instance of RandomForestRegressor
    model = RandomForestRegressor(max_depth=10, n_estimators=50, random_state=59)

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    predictions = model.predict(X_test)

    # Return the predictions
    return predictions