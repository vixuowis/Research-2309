from f00513_load_image_processor import *
def test_load_image_processor():
    checkpoint = "nvidia/mit-b0"
    image_processor = load_image_processor(checkpoint)

    # Test case 1
    assert isinstance(image_processor, AutoImageProcessor)

    # Test case 2
    assert image_processor.reduce_labels == True

    # Test case 3
    assert image_processor.checkpoint == checkpoint

test_load_image_processor()
