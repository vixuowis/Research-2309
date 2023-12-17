# requirements_file --------------------

!pip install -U PIL requests transformers

# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def geolocalize_image(image_url, possible_cities):
    # Load the 'geolocal/StreetCLIP' model and processor
    model = CLIPModel.from_pretrained('geolocal/StreetCLIP')
    processor = CLIPProcessor.from_pretrained('geolocal/StreetCLIP')

    # Fetch the image from the provided URL
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Process the texts (city names) and image using the CLIPProcessor
    inputs = processor(text=possible_cities, images=image, return_tensors='pt', padding=True)

    # Pass the input to the model
    outputs = model(**inputs)

    # Compute probabilities for different cities using softmax
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).tolist()[0]

    # Create a dictionary of cities and their respective probabilities
    city_probs = dict(zip(possible_cities, probs))
    return city_probs

# test_function_code --------------------

def test_geolocalize_image():
    print("Testing geolocalize_image function.")
    # Example image URL
    example_image_url = 'https://huggingface.co/geolocal/StreetCLIP/resolve/main/sanfrancisco.jpeg'
    # Possible cities to match the image
    example_cities = ['San Jose', 'San Diego', 'Los Angeles', 'Las Vegas', 'San Francisco']

    # Make a prediction using the geolocalize_image function
    prediction = geolocalize_image(example_image_url, example_cities)

    # Check if the result is a dictionary
    assert isinstance(prediction, dict), "The result should be a dictionary."

    # Check if the dictionary contains probabilities for each city
    assert all(city in prediction for city in example_cities), "The result should contain all the example cities."
    print("Testing geolocalize_image function completed successfully.")

# Run the test function
test_geolocalize_image()