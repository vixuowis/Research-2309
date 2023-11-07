from f00830_depth_estimation import *
def test_depth_estimation():
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    depth_values = depth_estimation(image_url)
    assert isinstance(depth_values, list)
    assert all(isinstance(depth, float) for depth in depth_values)

test_depth_estimation()
