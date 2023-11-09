def test_predict_link_building_strategy():
    """
    This function tests the predict_link_building_strategy function by using a small sample of data.
    """
    data = pd.read_csv('data.csv')
    predictions = predict_link_building_strategy(data)
    assert predictions is not None, 'No predictions were made.'
    assert len(predictions) == len(data), 'The number of predictions does not match the number of instances in the input data.'

test_predict_link_building_strategy()