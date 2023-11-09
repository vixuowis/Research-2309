# function_import --------------------

from keras.tab_transformer.TabTransformer import TabTransformer

# function_code --------------------

def classify_customer_segments(X_train, y_train, X_test):
    """
    This function uses the TabTransformer model from the keras-io/tab_transformer library to classify customer behavior into different segments.

    Args:
        X_train (DataFrame): The training data. It should contain both numerical and categorical features related to customer behavior.
        y_train (Series): The target values for the training data.
        X_test (DataFrame): The testing data. It should contain the same features as the training data.

    Returns:
        predictions (Series): The predicted segments for the testing data.
    """
    tab_transformer = TabTransformer.from_config()
    tab_transformer.fit(X_train, y_train)
    predictions = tab_transformer.predict(X_test)
    return predictions

# test_function_code --------------------

def test_classify_customer_segments():
    """
    This function tests the classify_customer_segments function by using a small sample of data.
    """
    # Create a small sample of data
    X_train = pd.DataFrame({'numerical': [1, 2, 3], 'categorical': ['A', 'B', 'C']})
    y_train = pd.Series([0, 1, 0])
    X_test = pd.DataFrame({'numerical': [2, 3, 4], 'categorical': ['B', 'C', 'D']})

    # Call the function with the sample data
    predictions = classify_customer_segments(X_train, y_train, X_test)

    # Check that the function returns a Series
    assert isinstance(predictions, pd.Series), 'The function should return a Series.'

    # Check that the length of the predictions matches the length of the testing data
    assert len(predictions) == len(X_test), 'The length of the predictions should match the length of the testing data.'

# call_test_function_code --------------------

test_classify_customer_segments()