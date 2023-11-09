def test_predict_carbon_emissions():
    """
    This function tests the predict_carbon_emissions function by using a sample dataset.
    """
    predictions = predict_carbon_emissions('test_data.csv')
    assert isinstance(predictions, list), 'The result should be a list.'
    assert all(isinstance(i, (int, float)) for i in predictions), 'Each prediction should be a number.'
    assert len(predictions) > 0, 'The list of predictions should not be empty.'

test_predict_carbon_emissions()