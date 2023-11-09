def test_predict_tip():
    # Test the predict_tip function
    prediction = predict_tip(39.42, 0, 0, 6, 0, 4)
    # Since we are dealing with a regression model, we cannot compare the prediction strictly.
    # Instead, we check if the prediction is within a reasonable range.
    assert 0 <= prediction <= 10

test_predict_tip()