def test_predict_vacation_success():
    """
    This function tests the predict_vacation_success function.
    It uses a sample data to test the function and checks if the output is as expected.
    """
    destination = 'Bali'
    accommodation = 'Hotel'
    travel_style = 'Solo'
    prediction = predict_vacation_success(destination, accommodation, travel_style)
    assert isinstance(prediction, int), 'The prediction should be an integer.'
    assert prediction in [0, 1], 'The prediction should be either 0 or 1.'

test_predict_vacation_success()