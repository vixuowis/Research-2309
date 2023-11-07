from f00828_object_detection import *
def test_object_detection():
    image_url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg'
    expected_output = [{'score': 0.9865, 'label': 'cat', 'box': {'xmin': 178, 'ymin': 154, 'xmax': 882, 'ymax': 598}}]
    assert object_detection(image_url) == expected_output

test_object_detection()
