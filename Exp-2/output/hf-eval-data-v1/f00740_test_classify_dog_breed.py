from datasets import load_dataset

# Function to test the classify_dog_breed function
# @param: None
# @return: None
def test_classify_dog_breed():
    # Load the test dataset
    dataset = load_dataset('huggingface/cats-image')

    # Select a sample image
    image = dataset['test']['image'][0]

    # Get the predicted breed
    predicted_breed = classify_dog_breed(image)

    # Assert that the predicted breed is a string
    assert isinstance(predicted_breed, str)

    # Print the predicted breed
    print('Predicted breed:', predicted_breed)

test_classify_dog_breed()