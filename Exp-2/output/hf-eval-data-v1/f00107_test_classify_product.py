def test_classify_product():
    '''
    This function tests the classify_product function by using a sample image from the 'imagenet-1k' dataset.
    '''
    # Load the 'imagenet-1k' dataset
    dataset = load_dataset('imagenet-1k')

    # Select a sample image from the dataset
    image = dataset['test']['image'][0]

    # Classify the sample image
    predicted_label = classify_product(image)

    # Check if the predicted label is a string (as it should be)
    assert isinstance(predicted_label, str)