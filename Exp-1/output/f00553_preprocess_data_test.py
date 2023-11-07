from f00553_preprocess_data import *
def test_preprocess_data():
    checkpoint = 'facebook/detr-resnet-50'
    image_processor = preprocess_data(checkpoint)

    # Test attributes
    assert image_processor.image_mean == [0.485, 0.456, 0.406]
    assert image_processor.image_std == [0.229, 0.224, 0.225]

    # Test instance type
    assert isinstance(image_processor, AutoImageProcessor)

    print('All tests passed!')

test_preprocess_data()
