def test_classify_animal_species():
    # Test the function with a sample image
    animal_image_path = 'test_image.jpg'  # Replace with the path of your test image
    predicted_species = classify_animal_species(animal_image_path)
    print(predicted_species)
    # Assert the function returns a string (the name of the species)
    assert isinstance(predicted_species, str)

test_classify_animal_species()