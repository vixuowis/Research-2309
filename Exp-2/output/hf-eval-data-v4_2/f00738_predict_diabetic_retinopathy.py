# requirements_file --------------------

!pip install -U transformers datasets tokenizers torch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def predict_diabetic_retinopathy(image_path):
    """
    Predicts whether the given image indicates diabetic retinopathy.

    Args:
        image_path (str): The path to the image file,

    Returns:
        dict: The prediction result with probabilities,

    Raises:
        Exception: If there is an error loading the model or processing the image.
    """
    try:
        image_classifier = pipeline('image-classification', 'martinezomg/vit-base-patch16-224-diabetic-retinopathy')
        result = image_classifier(image_path)
        return result
    except Exception as e:
        raise Exception(f'Error predicting diabetic retinopathy: {e}')

# test_function_code --------------------

def test_predict_diabetic_retinopathy():
    print("Testing started.")
    # Note: As we cannot load real images in this test function, we'll be using a placeholder 'eye.jpg'
    image_path = 'eye.jpg'  # Placeholder for image path

    # Testing case 1
    print("Testing case [1/1] started.")
    try:
        prediction = predict_diabetic_retinopathy(image_path)
        assert type(prediction) == list, "Prediction should be a list"
    except Exception as e:
        assert False, f"Test case [1/1] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_predict_diabetic_retinopathy()