def test_classify_device():
    # Test image paths
    test_image_paths = ['test_image1.jpg', 'test_image2.jpg', 'test_image3.jpg']
    # Expected device types
    expected_device_types = [0, 1, 2]
    # Test classify_device function
    for i, image_path in enumerate(test_image_paths):
        assert classify_device(image_path) == expected_device_types[i]