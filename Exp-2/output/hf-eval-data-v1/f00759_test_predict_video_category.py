def test_predict_video_category():
    """
    Test the function predict_video_category.
    """
    text = "A group of people playing football in a stadium"
    predicted_category, predicted_probability = predict_video_category(text)

    assert isinstance(predicted_category, str)
    assert 0 <= predicted_probability <= 1

    print(f'Test passed. Predicted category: {predicted_category}, probability: {predicted_probability}')

test_predict_video_category()