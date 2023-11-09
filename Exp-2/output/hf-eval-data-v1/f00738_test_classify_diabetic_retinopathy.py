# Test function for classify_diabetic_retinopathy
# Since no specific dataset is provided, we will use an online image for testing
# The test function will assert if the function is working as expected
def test_classify_diabetic_retinopathy():
    test_image_path = 'https://raw.githubusercontent.com/martinezomg/vit-base-patch16-224-diabetic-retinopathy/main/test_image.jpg'
    result = classify_diabetic_retinopathy(test_image_path)
    # Since the model's accuracy is not 100%, we cannot strictly compare the result
    # Instead, we will check if the result is in the expected format (a list of dictionaries)
    assert isinstance(result, list) and isinstance(result[0], dict) and 'label' in result[0] and 'score' in result[0]

test_classify_diabetic_retinopathy()