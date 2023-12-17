# requirements_file --------------------

!pip install -U ultralyticsplus==0.0.24 ultralytics==8.0.23 matplotlib

# function_import --------------------

from ultralyticsplus import YOLO, render_result
import matplotlib.pyplot as plt

# function_code --------------------

def detect_blood_cells(image_path):
    """
    Detect blood cells in the given microscopic image using a pre-trained YOLOv8 model.

    Args:
        image_path (str): The file path or URL to the microscopic blood sample image.

    Returns:
        plt.Figure: A matplotlib figure showing the image with detection boxes, labels, and scores.

    Raises:
        ValueError: If the image_path is not a string or None.
        RuntimeError: If the model fails to make predictions.
    """
    if not isinstance(image_path, str):
        raise ValueError('The image_path must be a string.')
    model = YOLO('keremberke/yolov8m-blood-cell-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(image_path)
    render = render_result(model=model, image=image_path, result=results[0])

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(render.render())
    ax.axis('off')
    plt.show()
    return fig

# test_function_code --------------------

def test_detect_blood_cells():
    print("Testing started.")
    sample_image_path = 'https://example.com/sample_blood_image.jpg'  # Replace with a real URL or file path

    # Test case 1: Valid image path
    print("Testing case [1/1] started.")
    try:
        fig = detect_blood_cells(sample_image_path)
        assert isinstance(fig, plt.Figure), f"Test case [1/1] failed: The result is not a matplotlib figure."
    except Exception as e:
        assert False, f"Test case [1/1] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_blood_cells()