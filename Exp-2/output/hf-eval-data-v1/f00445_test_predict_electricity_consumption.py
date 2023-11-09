def test_predict_electricity_consumption():
    """
    This function tests the predict_electricity_consumption function.
    It uses a sample dataset for testing.
    """
    # Load a sample dataset
    data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv')
    data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'electricity_consumption']

    # Call the function with the sample dataset
    prediction = predict_electricity_consumption(data)

    # Assert the prediction is a float
    assert isinstance(prediction, float), 'The prediction should be a float.'

    # Assert the prediction is not strictly equal to a specific number
    assert prediction != 100, 'The prediction should not be strictly equal to a specific number.'

test_predict_electricity_consumption()