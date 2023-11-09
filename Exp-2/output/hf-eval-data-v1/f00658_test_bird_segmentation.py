# Test function for bird_segmentation
# Uses an image from the COCO dataset
# The function is expected to return a predicted instance map
# The test function does not compare numbers strictly

def test_bird_segmentation():
    # Test image URL
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # Call the bird_segmentation function
    predicted_instance_map = bird_segmentation(url)
    # Check that the function returns a result
    assert predicted_instance_map is not None
    # Check that the result is a torch.Tensor
    assert isinstance(predicted_instance_map, torch.Tensor)

test_bird_segmentation()