from f00481_apply_image_transformations import *
def test_apply_image_transformations():
    image_processor = {
        'image_mean': [0.485, 0.456, 0.406],
        'image_std': [0.229, 0.224, 0.225],
        'size': {
            'shortest_edge': 256
        }
    }
    transforms = apply_image_transformations(image_processor)
    assert isinstance(transforms, object)
    assert hasattr(transforms, '__call__')


test_apply_image_transformations()
