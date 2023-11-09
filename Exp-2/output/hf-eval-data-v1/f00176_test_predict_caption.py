def test_predict_caption():
    # Test the predict_caption function
    # Note: The test is not strict on the exact caption as it can vary
    # We just check if the function returns a string (caption)
    caption = predict_caption("sample_image.jpg")
    assert isinstance(caption, str), "Caption should be a string"

test_predict_caption()