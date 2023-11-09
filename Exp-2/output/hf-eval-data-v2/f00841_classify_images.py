# function_import --------------------

from transformers import AutoModelForImageClassification, ImageFeatureExtractionMixin

# function_code --------------------

def classify_images(image_paths, categories=['category1', 'category2', 'category3']):
    """
    Classify images into various categories using a pre-trained model from Hugging Face Transformers.

    Args:
        image_paths (list): List of paths to the images to be classified.
        categories (list, optional): List of categories. Defaults to ['category1', 'category2', 'category3'].

    Returns:
        dict: A dictionary where keys are image paths and values are their corresponding categories.
    """
    model = AutoModelForImageClassification.from_pretrained('microsoft/swin-tiny-patch4-window7-224-bottom_cleaned_data')
    results = {}
    for image_path in image_paths:
        result = model.classify_image(image_path)
        results[image_path] = categories[result]
    return results

# test_function_code --------------------

def test_classify_images():
    """
    Test the classify_images function.
    """
    image_paths = ['path_to_image1', 'path_to_image2']
    categories = ['category1', 'category2', 'category3']
    results = classify_images(image_paths, categories)
    for image_path, category in results.items():
        assert category in categories, f'Error: {category} not in {categories}'

# call_test_function_code --------------------

test_classify_images()