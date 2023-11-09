def test_classify_co2_emissions():
    """
    Test the classify_co2_emissions function.
    """
    # Use a small subset of the dataset for testing
    test_data_path = 'test_CO2_emissions.csv'
    predictions = classify_co2_emissions(test_data_path)
    # Check that the function returns a list
    assert isinstance(predictions, list)
    # Check that the length of the list matches the number of rows in the test dataset
    test_data = pd.read_csv(test_data_path)
    assert len(predictions) == len(test_data)