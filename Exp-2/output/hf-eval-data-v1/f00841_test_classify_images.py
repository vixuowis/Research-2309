def test_classify_images():
    """
    Test the classify_images function.
    """
    # Define the image paths and categories
    image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
    categories = ['category1', 'category2', 'category3']

    # Call the function
    results = classify_images(image_paths, categories)

    # Check the results
    for image_path, category in results.items():
        print(f'Image {image_path} was classified as {category}.')
        assert category in categories

test_classify_images()