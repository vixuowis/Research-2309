def test_classify_security_camera_image():
    # Load a test dataset
    dataset = load_dataset('huggingface/cats-image')
    # Select a sample image from the dataset
    test_image = dataset['test']['image'][0]
    # Call the function with the test image
    predicted_label = classify_security_camera_image(test_image)
    # Assert that the function returns a string (the label)
    assert isinstance(predicted_label, str)
    # Print the predicted label for the test image
    print('Predicted label for test image:', predicted_label)

test_classify_security_camera_image()