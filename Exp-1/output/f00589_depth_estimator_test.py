from f00589_depth_estimator import *
def test_depth_estimator():
    # Create test image
    image = Image.new('RGB', (256, 256))

    # Call depth_estimator function
    predictions = depth_estimator(image)

    # Assert the output
    assert isinstance(predictions, torch.Tensor)


test_depth_estimator()
