# requirements_file --------------------

import subprocess

requirements = ["ultralyticsplus", "ultralytics"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from ultralyticsplus import YOLO, render_result
from PIL import Image
import requests

# function_code --------------------

def extract_table_from_image(image_url):
    """
    Extracts a table from the given image URL and visualizes the result.

    Args:
        image_url (str): URL to the image containing the table to be extracted.

    Returns:
        PIL.Image.Image: The rendered image with the table extraction results.

    Raises:
        ValueError: If the image URL is not valid.
        RuntimeError: If the model fails to process the image.
    """
    # Load the image from the URL
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        raise ValueError("Invalid image URL.") from e

    # Instantiate the YOLO model for table extraction
    model = YOLO('keremberke/yolov8n-table-extraction')

    # Set the model overrides
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    # Use the YOLO model to predict the table regions
    try:
        results = model.predict(image)
    except Exception as e:
        raise RuntimeError("Model failed to process image.") from e

    # Render the results and return the rendered image
    render = render_result(model=model, image=image, result=results[0])
    return render.show()

# test_function_code --------------------

def test_extract_table_from_image():
    from io import BytesIO
    print("Testing started.")

    # Example image URLs
    image_urls = [
        'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg',
        'https://github.com/ultralytics/yolov5/raw/master/data/images/bus.jpg',
        'InvalidURL'
    ]

    for i, image_url in enumerate(image_urls, start=1):
        print(f"Testing case [{i}/{len(image_urls)}] started.")
        try:
            result_image = extract_table_from_image(image_url)
            assert result_image, f"Test case [{i}/{len(image_urls)}] failed: No result image generated."
        except Exception as e:
            assert 'Invalid image URL' in str(e) or 'Model failed to process image' in str(e), f"Test case [{i}/{len(image_urls)}] failed with unexpected error: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_extract_table_from_image()