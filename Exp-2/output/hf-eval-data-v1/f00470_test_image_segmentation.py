def test_image_segmentation():
    '''
    This function tests the image_segmentation function.
    It uses an online image for testing.
    '''
    image_url = 'https://huggingface.co/datasets/shi-labs/oneformer_demo/blob/main/ade20k.jpeg'
    image_path = 'test_image.jpg'
    with open(image_path, 'wb') as f:
        f.write(requests.get(image_url).content)

    for task in ['semantic', 'instance', 'panoptic']:
        predicted_map = image_segmentation(image_path, task)
        assert isinstance(predicted_map, dict), f'Error in {task} segmentation'

    os.remove(image_path)

test_image_segmentation()